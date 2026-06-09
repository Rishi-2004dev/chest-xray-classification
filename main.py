#this is the backend file and this creates a API for my model so that it could interact with web.
import io
import os
import logging
import asyncio
import threading
from contextlib import asynccontextmanager

import numpy as np
from PIL import Image, ImageStat
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf

# =====================================================================
# Configuration & Constants
# =====================================================================

MODEL_PATH = "cnn_xray_model.keras"
CLASS_NAMES = ["NORMAL", "PNEUMONIA", "TUBERCULOSIS"]

MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png"}

# WARNING: Medically Unsafe Heuristic.
# If True, rejects images with high color variance (e.g., leaves, natural photos).
# Kept False by default because valid X-rays often contain colored text markers from hospital PACS systems.
STRICT_OOD_REJECTION = False

# Configure standard Python logging for production visibility
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global state
ml_dependencies = {}
# TensorFlow predict() is not thread-safe. This lock prevents race conditions during concurrent requests.
inference_lock = threading.Lock()

# =====================================================================
# Lifespan Management (Startup & Shutdown)
# =====================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles application startup and shutdown events.
    Fails fast if the model cannot be loaded, preventing runtime errors later.
    """
    logger.info("Initializing application startup sequence...")

    # 1. Enable TensorFlow GPU Memory Growth
    physical_gpus = tf.config.list_physical_devices('GPU')
    if physical_gpus:
        try:
            for gpu in physical_gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"Enabled GPU memory growth for {len(physical_gpus)} GPU(s).")
        except RuntimeError as e:
            logger.error(f"Failed to set GPU memory growth: {e}")
    else:
        logger.info("No visible GPUs found. Running on CPU.")

    # 2. Load the TensorFlow Model
    if not os.path.exists(MODEL_PATH):
        logger.critical(f"Model file not found at {MODEL_PATH}. Server cannot start.")
        raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")

    try:
        logger.info(f"Loading model from {MODEL_PATH}...")
        model = tf.keras.models.load_model(MODEL_PATH)
        ml_dependencies["model"] = model
        logger.info("Model loaded successfully.")
        
        # 3. Model Warmup
        logger.info("Running dummy prediction to warm up the model graph...")
        dummy_input = np.zeros((1, 224, 224, 3), dtype=np.float32)
        with inference_lock:
            model.predict(dummy_input, verbose=0)
        logger.info("Model warmup complete.")

    except Exception as e:
        logger.critical(f"Failed to load the model: {str(e)}")
        raise RuntimeError(f"Model loading failed: {str(e)}")

    # Yield control back to FastAPI to start accepting requests
    yield

    # Teardown logic
    logger.info("Shutting down application. Clearing ML resources.")
    ml_dependencies.clear()

# =====================================================================
# FastAPI Application Initialization
# =====================================================================

app = FastAPI(
    title="Chest X-Ray Classification API",
    description="Production API for medical imaging inference using TensorFlow.",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Configuration
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)

# =====================================================================
# Core ML Inference Logic (Synchronous / CPU Bound)
# =====================================================================

def process_and_predict(image_bytes: bytes) -> dict:
    """
    Synchronous function executing CPU-bound image processing and TF inference.
    Designed to be run in a separate thread pool.
    """
    # -- 1. Safe Image Parsing --
    try:
        img = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        logger.warning(f"Failed to parse image bytes: {e}")
        raise ValueError("Invalid image file format or corrupted file.")

    # -- 2. OOD (Out of Distribution) Check --
    img_rgb_check = img.convert("RGB")
    stat = ImageStat.Stat(img_rgb_check)
    r_var, g_var, b_var = stat.var
    
    color_variance = abs(r_var - g_var) + abs(g_var - b_var) + abs(b_var - r_var)
    
    if color_variance > 1000:
        warning_msg = "Image contains significant color variance. It may not be a valid X-ray."
        if STRICT_OOD_REJECTION:
            logger.warning(f"OOD Rejected: {warning_msg}")
            raise ValueError(warning_msg)
        else:
            logger.info(f"OOD Warning (Bypassed): {warning_msg}")

    # -- 3. Exact Training Preprocessing Pipeline --
    try:
        img = img.convert("RGB")
        # Explicitly force BILINEAR to match TF/Keras defaults and prevent domain shift
        img = img.resize((224, 224), resample=Image.BILINEAR)
        
        img_array = np.array(img, dtype=np.float32)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
    except Exception as e:
        logger.error(f"Preprocessing pipeline failed: {e}")
        raise RuntimeError("Internal error during image preprocessing.")

    # -- 4. Thread-Safe Inference --
    try:
        model = ml_dependencies["model"]
        
        # Lock required because model.predict is not inherently thread-safe in concurrent requests
        with inference_lock:
            predictions = model.predict(img_array, verbose=0)[0]
        
        # Assumption: The saved Keras model contains a Softmax layer.
        # If it output logits, tf.nn.softmax(predictions).numpy() would be required here.
        predicted_idx = int(np.argmax(predictions))
        confidence_score = float(np.max(predictions))
        
        return {
            "prediction": CLASS_NAMES[predicted_idx],
            "confidence": f"{confidence_score * 100:.2f}%"
        }
    except Exception as e:
        logger.error(f"Model prediction failed: {e}")
        raise RuntimeError("Internal error during model inference.")

# =====================================================================
# API Endpoints
# =====================================================================

@app.post("/predict", status_code=status.HTTP_200_OK)
async def predict_xray(file: UploadFile = File(...)):
    """
    Accepts an X-ray image upload, validates the file, and returns a diagnosis prediction.
    """
    # 1. MIME Type Validation (415 Client Error)
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{file.content_type}'. Allowed types are: {', '.join(ALLOWED_MIME_TYPES)}"
        )

    # 2. File Size Limit (413 Client Error)
    image_bytes = await file.read()
    if len(image_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds the {MAX_FILE_SIZE_BYTES // (1024 * 1024)}MB size limit."
        )

    # 3. Offload CPU-bound inference to a background thread
    try:
        result = await asyncio.to_thread(process_and_predict, image_bytes)
    except ValueError as ve:
        # 400 Bad Request: Corrupted image or strictly rejected OOD
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        # 500 Internal Server Error: Preprocessing or TF failures
        logger.error(f"Inference processing failed: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Inference processing failed.")

    # 4. JSON Response
    return {
        "filename": file.filename,
        "prediction": result["prediction"],
        "confidence": result["confidence"]
    }

@app.get("/healthz")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    # Execute via: python main.py
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

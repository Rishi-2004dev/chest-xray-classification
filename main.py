#this is the backend file and this creates a API for my model so that it could interact with web.
import io
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile
import tensorflow as tf

# ==========================================
# Prevent TF from eating all VRAM
# ==========================================
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("[SYSTEM] GPU Memory Growth Enabled (No more greedy VRAM consumption)")
    except RuntimeError as e:
        print(f"[WARNING] Memory growth could not be set: {e}")

# Initialize the API
app = FastAPI(title="X-Ray Diagnostic API", version="1.0")

# Load the model strictly ONCE when the server starts
print("[SYSTEM] Loading Machine Learning Model...")
try:
    model = tf.keras.models.load_model("cnn_xray_model.keras")
    print("[SUCCESS] Model Loaded!")
except Exception as e:
    print(f"[FATAL ERROR] Model failed to load: {e}")

# Exact same mapping you used in training
CLASS_NAMES = ["NORMAL", "PNEUMONIA", "TUBERCULOSIS"]

def preprocess_image(image_bytes):
    """
    CRITICAL: This function must exactly replicate the tf.data pipeline
    from your training script.
    """
    # Read bytes and convert to RGB
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # Resize to exactly 224x224
    img = img.resize((224, 224))
    
    # Convert to Numpy Array
    img_array = np.array(img)
    
    # Normalize (Just like img / 255.0 in your pipeline)
    img_array = img_array / 255.0
    
    # Add batch dimension. Model expects (None, 224, 224, 3) -> (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

@app.post("/predict")
async def predict_xray(file: UploadFile = File(...)):
    # 1. Receive the file bytes
    image_bytes = await file.read()
    
    # 2. Transform the image
    processed_tensor = preprocess_image(image_bytes)
    
    # 3. Feed forward through the CNN
    predictions = model.predict(processed_tensor)
    
    # 4. Extract the highest probability
    predicted_index = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_index])
    
    # 5. Return JSON payload
    return {
        "filename": file.filename,
        "prediction": CLASS_NAMES[predicted_index],
        "confidence": f"{round(confidence * 100, 2)}%"
    }
#this code is correct but this model is not efficient for my smaller dataset as my simple 3 block model is giving more accuracy then densnet121.
#so iam not going to use this model for my project but anyone else who are interested can use this and can work upon this.
# ============================================================
# IMPORTS
# ============================================================
import os
import random
from pathlib import Path
from collections import defaultdict, Counter
from PIL import Image, ImageOps, ExifTags
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import DenseNet121

# ============================================================
# DATASET CONFIGURATION
# ============================================================
# Base directory of dataset
base_dir = Path(
    "/data/rishi and dev/rishi's coding stuff/coding stuff/python/python_ptojects/playing with deep learning and new datasets /combined-unknown-pneumonia-and-tuberculosis"
)

# Folder structure
dirs = ["NORMAL", "PNEUMONIA", "TUBERCULOSIS"]
dir_type = ["train", "test", "val"]

# Valid image extensions
valid_ext = {".jpg", ".png", ".jpeg"}

# ==============================================================
# file path + labels
#===============================================================
train_paths, train_labels = [], []
test_paths, test_labels = [], []
val_paths, val_labels = [], []
for split in dir_type:
    for label in dirs:
        current_dir = base_dir / "data" / split / label

        if not current_dir.exists():
            print(f"[SKIP] {current_dir} does not exist")
            continue

        for p in current_dir.iterdir():
            if not p.is_file():
                continue
            if p.suffix.lower() not in valid_ext:
                continue

            if split == "train":
                train_paths.append(str(p))
                train_labels.append(label)

            elif split == "test":
                test_paths.append(str(p))
                test_labels.append(label)

            elif split == "val":
                val_paths.append(str(p))
                val_labels.append(label)

print("trian size:", len(train_paths))
print("test_size:", len(test_paths))
print("Val size  :", len(val_paths))

# ============================================================
# LABEL ENCODING
# ============================================================
'''
Goal:
Convert string labels → numeric labels

Why:
- TensorFlow models do not understand strings
- Required for training
'''

label_to_index = {
    "NORMAL": 0,
    "PNEUMONIA": 1,
    "TUBERCULOSIS": 2
}

# ---- Encode train labels ----
train_labels_encoded = []
for l in train_labels:
    train_labels_encoded.append(label_to_index[l])

# ---- Encode test labels ----
test_labels_encoded = []
for l in test_labels:
    test_labels_encoded.append(label_to_index[l])

# ---- Encode val labels ----
val_labels_encoded = []
for l in val_labels:
    val_labels_encoded.append(label_to_index[l])


# ---- Sanity Check ----
#print("\nSample labels (before):", train_labels[:5])
#print("Sample labels (after):", train_labels_encoded[:5])
#
#print("\nUnique encoded labels:")
#print(set(train_labels_encoded))

# ============================================================
# CREATE TF.DATA DATASET (PART 1)
# ============================================================
'''
Goal:
Convert python lists → TensorFlow dataset

Why:
- TensorFlow models need dataset, not lists
- Enables batching, preprocessing, pipeline
'''
#----global shuffle---
train_pairs = list(zip(train_paths, train_labels_encoded))
random.shuffle(train_pairs)
train_paths, train_labels_encoded = zip(*train_pairs) #unpack list into separate arguments

test_pairs = list(zip(test_paths, test_labels_encoded))
random.shuffle(test_pairs)
test_paths, test_labels_encoded = zip(*test_pairs)

val_pairs = list(zip(val_paths, val_labels_encoded))
random.shuffle(val_pairs)
val_paths, val_labels_encoded = zip(*val_pairs)
## ---- Create train dataset ----
train_dataset = tf.data.Dataset.from_tensor_slices(
    (list(train_paths), list(train_labels_encoded))
)

# ---- Create test dataset ----
test_dataset = tf.data.Dataset.from_tensor_slices(
    (list(test_paths), list(test_labels_encoded))
)

# ---- Create val dataset ----
val_dataset = tf.data.Dataset.from_tensor_slices(
    (list(val_paths), list(val_labels_encoded))
)


# ---- Sanity Check ----
#print("\nSample from train_dataset:\n")
#
#for x in train_dataset.take(3):
#    print(x)
#
# ============================================================
# (PART 2): LOAD + PREPROCESS
# ============================================================

def preprocess_tf(path, label):

    # ---- read file (raw bytes) ----
    img = tf.io.read_file(path)

    # ---- decode + force RGB ----
    img = tf.image.decode_image(img, channels=3)

    # ---- fix shape issue ----
    img.set_shape([None, None, 3])

    # ---- resize ----
    img = tf.image.resize(img, (224, 224))

    # ---- normalize ----
    img = img / 255.0

    return img, label


# ---- apply preprocessing ----
train_dataset = train_dataset.map(preprocess_tf,num_parallel_calls=tf.data.AUTOTUNE)
test_dataset  = test_dataset.map(preprocess_tf,num_parallel_calls=tf.data.AUTOTUNE)
val_dataset   = val_dataset.map(preprocess_tf,num_parallel_calls=tf.data.AUTOTUNE)


# ============================================================
# STEP 3 (PART 3): SHUFFLE + BATCH + PREFETCH
# ============================================================

# ---- shuffle only train ----
train_dataset = train_dataset.shuffle(buffer_size=1000)

# ---- batch ----
train_dataset = train_dataset.batch(32)
test_dataset  = test_dataset.batch(32)
val_dataset   = val_dataset.batch(32)

# ---- prefetch (performance boost) ----
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
test_dataset  = test_dataset.prefetch(tf.data.AUTOTUNE)
val_dataset   = val_dataset.prefetch(tf.data.AUTOTUNE)


# ============================================================
# FINAL SANITY CHECK
# ============================================================

#for img, label in train_dataset.take(1):
#    print("\nImage batch shape:", img.shape)
#    print("Label batch:", label)

# ============================================================
# STEP 4: BUILD DENSENET121 TRANSFER LEARNING MODEL
# ============================================================

# 1. Load the pre-trained DenseNet base
base_model = DenseNet121(
    weights='imagenet', 
    include_top=False, 
    input_shape=(224, 224, 3)
)
print(len(base_model.layers))
#Freeze the base model
base_model.trainable=False

# 1. Aggressive Data Augmentation to destroy the JPEG/PNG Shortcut
data_augmentation = keras.Sequential([
    layers.RandomRotation(factor=0.05),
    layers.RandomTranslation(height_factor=0.05, width_factor=0.05),
    layers.RandomZoom(height_factor=0.05),
    layers.RandomContrast(factor=0.2),
    layers.GaussianNoise(stddev=0.05) 
], name="augmentation_block")

# 2. Model Architecture
inputs = layers.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = base_model(x, training=False)

# custom classification Head 
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.5)(x) # Added to prevent the new dense layer from overfitting
outputs = layers.Dense(3, activation='softmax')(x)

model = keras.Model(inputs, outputs)

print("\n--- Model Summary ---")
model.summary()
# ============================================================
# COMPILE AND TRAIN MODEL
# ============================================================

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 3. Early Stopping to prevent the Epoch 5-10 memory burn
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# 4. Class Weights to handle the imbalance
# (Calculated from your previous counts: Total 11840 / (3 * Class_Count))
class_weights = {
    0: 0.846,  # NORMAL
    1: 1.092,  # PNEUMONIA
    2: 1.107   # TUBERCULOSIS
}

print("\n--- Training DenseNet121 Transfer Learning Model ---")
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=15, 
    callbacks=[early_stopping],
    class_weight=class_weights
)

# ============================================================
# STEP 5: EVALUATE BASELINE ON TEST SET
# ============================================================
#print("\n--- Evaluating Baseline on Test Data ---")
#test_loss, test_acc = model.evaluate(test_dataset)
#print(f"Final Test Accuracy: {test_acc:.4f}")
#print(f"Final Test Loss: {test_loss:.4f}")

# ============================================================
# STEP 6: FINE-TUNING (THE THAW)
# ============================================================
print("\n" + "="*50)
print("PHASE 2: FINE-TUNING THE TOP LAYERS OF DENSENET")
print("="*50 + "\n")

# 1. Unfreeze the entire base model
base_model.trainable = True

# 2. Re-freeze the bottom layers, leave only the top block thawed
# DenseNet121 has 427 layers. We will unfreeze the last 30 layers.
for layer in base_model.layers[:-10]:
    layer.trainable = False

print(f"Total layers in base model: {len(base_model.layers)}")
print("Layers unfrozen for fine-tuning: 10")

# 3. RECOMPILE the model with a MICROSCOPIC learning rate
# This is critical. If you use 0.001 here, you destroy the model.
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5), # 0.00001
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 4. CREATE A NEW EARLY STOPPING CALLBACK FOR PHASE 2
# This resets the memory so it doesn't instantly kill the fine-tuning
fine_tune_early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)
# 5. Train the model again (it will resume from where it left off)
fine_tune_epochs = 40 
total_epochs = 15 + fine_tune_epochs # 15 from previous run + 10 new ones

history_fine = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=total_epochs,
    initial_epoch=len(history.epoch), # Start from the exact epoch where it stopped
    callbacks=[fine_tune_early_stopping], # Keep early stopping active
    class_weight=class_weights
)
# ============================================================
# STEP 7: FINAL EVALUATION AFTER FINE-TUNING
# ============================================================
print("\n--- Evaluating Fine-Tuned Model on Test Data ---")
final_test_loss, final_test_acc = model.evaluate(test_dataset)
print(f"ULTIMATE Test Accuracy: {final_test_acc:.4f}")
print(f"ULTIMATE Test Loss: {final_test_loss:.4f}")
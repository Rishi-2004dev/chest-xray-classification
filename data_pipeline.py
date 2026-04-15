#this is the main model that i am going to use as this gives more accuracy than densnet121. 
#this gives 90% accuracy, but in futher phases of this project and upgradation of this project we will try some different model to get the best highest accuracy than this basic 3 layer cnn network.
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
#pairs = list(zip(train_paths, train_labels))
#sample = random.sample(pairs, 10)
#for path, label in sample:
#    print(f"path is: {path} and label is {label}")

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
# STEP 4: BUILD ROBUST BASELINE CNN MODEL
# ============================================================

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

# Augmentation ONLY runs during training
x = data_augmentation(inputs)

# Block 1
x = layers.Conv2D(32, (3,3), activation='relu')(x)
x = layers.MaxPooling2D()(x)

# Block 2
x = layers.Conv2D(64, (3,3), activation='relu')(x)
x = layers.MaxPooling2D()(x)

# Block 3
x = layers.Conv2D(128, (3,3), activation='relu')(x)
x = layers.MaxPooling2D()(x)

# Head (Fixed: Replaced Flatten with GlobalAveragePooling2D)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
outputs = layers.Dense(3, activation='softmax')(x)

model = keras.Model(inputs, outputs)

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
    patience=4,
    restore_best_weights=True
)

# 4. Class Weights to handle the imbalance
# (Calculated from your previous counts: Total 11840 / (3 * Class_Count))
class_weights = {
    0: 0.846,  # NORMAL
    1: 1.092,  # PNEUMONIA
    2: 1.107   # TUBERCULOSIS
}

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
print("\n--- Evaluating Baseline on Test Data ---")
test_loss, test_acc = model.evaluate(test_dataset)
print(f"Final Test Accuracy: {test_acc:.4f}")
print(f"Final Test Loss: {test_loss:.4f}")

# ============================================================
# STEP 6: SAVE MODEL FOR PRODUCTION DEPLOYMENT
# ============================================================
print("\n--- Saving Model for Deployment ---")
model.save("cnn_xray_model.keras")
print("[SUCCESS] Model saved as cnn_xray_model.keras")
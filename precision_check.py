#this python script only calc recall, F1 and precision for my dataset.
import os
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print("[SYSTEM] Loading Test Data and Model...")

# 1. Re-create ONLY the Test Dataset (Copying your exact logic)
base_dir = Path("/data/rishi and dev/rishi's coding stuff/coding stuff/python/python_ptojects/playing with deep learning and new datasets /combined-unknown-pneumonia-and-tuberculosis")

test_paths, test_labels = [], []
for label in ["NORMAL", "PNEUMONIA", "TUBERCULOSIS"]:
    current_dir = base_dir / "data" / "test" / label
    for p in current_dir.iterdir():
        if p.is_file() and p.suffix.lower() in {".jpg", ".png", ".jpeg"}:
            test_paths.append(str(p))
            test_labels.append(label)

label_to_index = {"NORMAL": 0, "PNEUMONIA": 1, "TUBERCULOSIS": 2}
test_labels_encoded = [label_to_index[l] for l in test_labels]

test_dataset = tf.data.Dataset.from_tensor_slices((list(test_paths), list(test_labels_encoded)))

def preprocess_tf(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, (224, 224))
    img = img / 255.0
    return img, label

# VERY IMPORTANT: shuffle=False so we can match predictions to true labels
test_dataset = test_dataset.map(preprocess_tf, num_parallel_calls=tf.data.AUTOTUNE).batch(32).prefetch(tf.data.AUTOTUNE)

# 2. Load the Saved Model
model = tf.keras.models.load_model("cnn_xray_model.keras")

# 3. Generate Predictions
print("\n[SYSTEM] Running predictions on test data (This might take a minute)...")
y_true = []
y_pred_probs = []

for images, labels in test_dataset:
    y_true.extend(labels.numpy())
    preds = model.predict(images, verbose=0)
    y_pred_probs.extend(preds)

y_true = np.array(y_true)
y_pred = np.argmax(y_pred_probs, axis=1)

# 4. Generate Classification Report
class_names = ["NORMAL", "PNEUMONIA", "TUBERCULOSIS"]
print("\n" + "="*50)
print("CLASSIFICATION REPORT (Precision, Recall, F1)")
print("="*50)
print(classification_report(y_true, y_pred, target_names=class_names))

# 5. Plot Confusion Matrix
print("\n[SYSTEM] Generating Confusion Matrix...")
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.ylabel('TRUE (Actual Disease)')
plt.xlabel('PREDICTED (What the Model Said)')
plt.savefig('confusion_matrix.png')
print("[SUCCESS] Confusion Matrix saved as 'confusion_matrix.png' in your folder.")
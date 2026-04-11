import tensorflow as tf
import matplotlib.pyplot as plt
import random
from pathlib import Path

# 1. Your exact path configuration
base_dir = Path(
    "/data/rishi and dev/rishi's coding stuff/coding stuff/python/"
    "python_ptojects/playing with deep learning and new new datasets /"
    "combined-unknown-pneumonia-and-tuberculosis"
)

# 2. Programmatically build the target directory
tb_dir = base_dir / "data" / "train" / "TUBERCULOSIS"

# Verify it exists before trying to load images
if not tb_dir.exists():
    print(f"ERROR: Directory does not exist! Checked path: {tb_dir}")
    exit()

# 3. Grab all PNGs
tb_images = list(tb_dir.glob("*.png"))

if len(tb_images) < 3:
    print(f"ERROR: Not enough PNG images found in {tb_dir}. Found: {len(tb_images)}")
    exit()

# 4. Select 3 random images
sample_paths = random.sample(tb_images, 3)

plt.figure(figsize=(12, 12))

for i, img_path in enumerate(sample_paths):
    # Load original
    raw_img = tf.io.read_file(str(img_path))
    original_img = tf.image.decode_image(raw_img, channels=3)
    
    # Crush to 224
    crushed_img = tf.image.resize(original_img, (224, 224))
    crushed_img = tf.cast(crushed_img, tf.uint8)
    
    # Plot Original
    plt.subplot(3, 2, 2*i + 1)
    plt.title(f"Original TB (High Res)\nShape: {original_img.shape}")
    plt.imshow(original_img)
    plt.axis('off')
    
    # Plot Crushed
    plt.subplot(3, 2, 2*i + 2)
    plt.title("Crushed (224x224)")
    plt.imshow(crushed_img)
    plt.axis('off')

plt.tight_layout()
# Save the plot as an image file instead of trying to open a window
output_path = base_dir / "tb_visual_test.png"
plt.savefig(output_path, dpi=300)
print(f"\n[SUCCESS] Image saved to: {output_path}")

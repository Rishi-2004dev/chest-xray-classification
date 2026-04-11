# ============================================================
# IMPORTS
# ============================================================
import os
from pathlib import Path
from collections import defaultdict, Counter
from PIL import Image, ImageOps, ExifTags
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf


# ============================================================
# DATASET CONFIGURATION
# ============================================================
# Base directory of dataset
base_dir = Path(
    "/data/rishi and dev/rishi's coding stuff/coding stuff/python/"
    "python_ptojects/playing with deep learning and new new datasets /"
    "combined-unknown-pneumonia-and-tuberculosis"
)

# Folder structure
dirs = ["NORMAL", "PNEUMONIA", "TUBERCULOSIS"]
dir_type = ["train", "test", "val"]

# Valid image extensions
valid_ext = {".jpg", ".png", ".jpeg"}


# ============================================================
# SECTION 1: FILE VALIDATION + BASIC DATASET ANALYSIS
# ============================================================
'''
This section checks:
1. Total files in each folder
2. Extension distribution (.jpg, .png, .jpeg)
3. Corrupt files (zero byte / missing)
4. Image mode distribution (RGB / L / P etc.)

IMPORTANT FINDING:
- Different classes have different mode distributions
- This leads to "Label Leakage"

Example:
- If grayscale → model predicts pneumonia

Fix later:
→ Convert ALL images to RGB
'''

for split in dir_type:
    for label in dirs:
        current_dir = base_dir / "data" / split / label
        if not current_dir.exists():
            print(f"[SKIP] {current_dir} does not exist")
            continue
        image_paths = [
            p for p in current_dir.iterdir()
            if p.is_file() and p.suffix.lower() in valid_ext
        ]
        print(f"\nDirectory: {split}/{label}")
        print("Total files:", len(image_paths))
        # ---- Extension Count ----
        count = {".jpeg": 0, ".png": 0, ".jpg": 0}
        for p in image_paths:
            count[p.suffix.lower()] += 1
        print(" .jpeg:", count[".jpeg"])
        print(" .png :", count[".png"])
        print(" .jpg :", count[".jpg"])
        # ---- Corrupt File Check ----
        bad_files = 0
        zero_byte_files = 0
        for img_path in current_dir.iterdir():
            if not img_path.is_file():
                continue
            if img_path.suffix.lower() not in valid_ext:
                continue
            if not img_path.exists():
                bad_files += 1
            elif img_path.stat().st_size == 0:
                zero_byte_files += 1
        print(f"{split}/{label} → bad_files={bad_files}, zero_byte_files={zero_byte_files}")
        # ---- Mode Distribution ----
        file_type = {"1": 0, "L": 0, "P": 0, "RGB": 0, "RGBA": 0}
        for p in image_paths:
            try:
                with Image.open(p) as img:
                    mode = img.mode
                    if mode in file_type:
                        file_type[mode] += 1
                    else:
                        print(f"[UNKNOWN MODE] {mode} in {p}")
            except Exception as e:
                print(f"[BROKEN IMAGE] {p} : {e}")
        print("\nImage mode distribution:")
        for mode, count in file_type.items():
            print(f"{mode}: {count}")


# ============================================================
# SECTION 2: EXIF ORIENTATION CHECK
# ============================================================
'''
Goal:
Check if any image depends on EXIF orientation

Why:
- Some images are stored rotated
- EXIF tells viewer how to display them
- ML models DO NOT read EXIF → potential bug

Conclusion from your dataset:
→ rotated = 0 → EXIF not important
'''

# Find Orientation tag ONCE (constant)
#ORIENTATION_TAG = None
#for k, v in ExifTags.TAGS.items():
#    if v == "Orientation":
#        ORIENTATION_TAG = k
#        break
#
#
#def get_exif_orientation(path):
#    try:
#        with Image.open(path) as img_ori:
#            exif = img_ori.getexif()
#            if not exif:
#                return None
#            return exif.get(ORIENTATION_TAG, None)
#    except Exception:
#        return None
#
#
## ============================================================
## SECTION 3: IMAGE SIZE COLLECTION
## ============================================================
#'''
#Goal:
#Store width and height of every image (per folder)
#
#Why:
#- Detect size distribution
#- Identify leakage (size → class)
#
#Finding:
#- NORMAL & TB → 512 + large scanner images
#- PNEUMONIA → mid-size images
#
#Fix later:
#→ Resize ALL images to 512×512
#'''
#
#sizes = defaultdict(lambda: {"widths": [], "heights": []})
#exif_stats = {}
#
#for split in dir_type:
#    for label in dirs:
#        current_dir = base_dir / "data" / split / label
#
#        if not current_dir.exists():
#            print(f"[SKIP] {current_dir} does not exist")
#            continue
#
#        image_paths = [
#            p for p in current_dir.iterdir()
#            if p.is_file() and p.suffix.lower() in valid_ext
#        ]
#
#        key = f"{split}/{label}"
#
#        # ---- Size Extraction ----
#        for img_path in image_paths:
#            try:
#                with Image.open(img_path) as img:
#                    w, h = img.size
#                    sizes[key]["widths"].append(w)
#                    sizes[key]["heights"].append(h)
#            except Exception as e:
#                print(f"[SKIP IMAGE] {img_path} {e}")
#
#        # ---- EXIF Stats (per folder) ----
#        total = 0
#        with_orientation = 0
#        rotated = 0
#
#        for img_ in image_paths:
#            total += 1
#            orientation = get_exif_orientation(img_)
#
#            if orientation is not None:
#                with_orientation += 1
#                if orientation in {3, 6, 8}:
#                    rotated += 1
#
#        exif_stats[key] = {
#            "total": total,
#            "with_orientation": with_orientation,
#            "rotated": rotated
#        }
#
#
## ============================================================
## SECTION 4: GLOBAL SIZE DISTRIBUTION
## ============================================================
#print("\nCount of image resolutions for the whole dataset:")
#size_counter = Counter()
#
#for key, data in sizes.items():
#    for w, h in zip(data["widths"], data["heights"]):
#        size_counter[(w, h)] += 1
#
#for (w, h), cnt in size_counter.most_common(25):
#    print(f"{w} x {h} : {cnt}")
#
#
## ============================================================
## SECTION 5: PER-FOLDER SIZE DISTRIBUTION
## ============================================================
#size_tables = {}
#
#for key, data in sizes.items():
#    c = Counter()
#    for w, h in zip(data["widths"], data["heights"]):
#        c[(w, h)] += 1
#    size_tables[key] = c
#
#for folder, counter in size_tables.items():
#    print(f"\nFolder: {folder}")
#    print("-" * 30)
#    for (w, h), cnt in counter.most_common(10):
#        print(f"{w} x {h} : {cnt}")
#
#
## ============================================================
## SECTION 6: EXIF REPORT
## ============================================================
#print("\nEXIF ORIENTATION REPORT")
#print("-" * 50)
#
#for k, v in exif_stats.items():
#    print(
#        f"{k:20} | "
#        f"total={v['total']:4} | "
#        f"orientation_tag={v['with_orientation']:4} | "
#        f"rotated={v['rotated']:4}"
#    )
#
#
# ============================================================
# SECTION 7: MD5 DUPLICATE DETECTION
# ============================================================

#import hashlib
#
#def get_md5(path):
#    with open(path, "rb") as f:
#        return hashlib.md5(f.read()).hexdigest()
#
#
#hash_map = {}
#duplicates = []
#total_images = 0   # 🔥 FIX
#
#for split in dir_type:
#    for label in dirs:
#        current_dir = base_dir / "data" / split / label
#
#        if not current_dir.exists():
#            print(f"[SKIP] {current_dir} does not exist")
#            continue
#
#        image_paths = [
#            p for p in current_dir.iterdir()
#            if p.is_file() and p.suffix.lower() in valid_ext
#        ]
#
#        for p in image_paths:
#            try:
#                total_images += 1   # 🔥 COUNT HERE
#
#                h = get_md5(p)
#
#                if h in hash_map:
#                    duplicates.append((hash_map[h], p))
#                else:
#                    hash_map[h] = p
#
#            except Exception as e:
#                print(f"[ERROR] {p} → {e}")
#
#
## ---- RESULT ----
#print("\nTotal images checked:", total_images)
#print("Total duplicates found:", len(duplicates))
#
#print("\nSome duplicate pairs:\n")
#
#for original, duplicate in duplicates[:42]:
#    print("ORIGINAL :", original)
#    print("DUPLICATE:", duplicate)
#    print("-" * 40)
#
#---- now removing duplicated----
#to_remove=[]
#for original, duplicate in duplicates:
#    orig=str(original)
#    dup=str(duplicate)
#    # ---- TRAIN vs TEST ----
#    if "/train/" in orig and "/test/" in dup:
#        to_remove.append(dup)
#    elif "/train/" in dup and "/test/" in orig:
#        to_remove.append(orig)
#
#    # ---- TRAIN vs VAL ----
#    elif "/train/" in orig and "/val/" in dup:
#        to_remove.append(dup)
#    elif "/train/" in dup and "/val/" in orig:
#        to_remove.append(orig)
#
#    # ---- TEST vs VAL ----
#    elif "/test/" in orig and "/val/" in dup:
#        to_remove.append(dup)
#    elif "/test/" in dup and "/val/" in orig:
#        to_remove.append(orig)
#
#    # ---- SAME SPLIT ----
#    else:
#        to_remove.append(dup)
#
#
## ---- REMOVE FILES ----
#for path in to_remove:
#    try:
#        os.remove(path)
#        print("Removed:", path)
#    except Exception as e:
#        print("Error:", path, e)
#


# ============================================================
# SECTION 8: OPTIONAL VISUALIZATION (HEXBIN)
# ============================================================
'''
This shows density of image sizes using log scale
Useful for:
- spotting dominant resolution
- identifying clusters
'''

# all_widths = []
# all_heights = []

# for data in sizes.values():
#     all_widths.extend(data["widths"])
#     all_heights.extend(data["heights"])

# plt.figure(figsize=(6, 6))
# plt.hexbin(all_widths, all_heights, gridsize=50, bins="log")
# plt.colorbar(label="log(count)")
# plt.xlabel("Width (pixels)")
# plt.ylabel("Height (pixels)")
# plt.title("Overall Image Size Distribution")
# plt.savefig("size_distribution.png", dpi=200, bbox_inches="tight")
# plt.close()



































#the code below this is 100% correct and structured version of this is given above.
#import os
#import pathlib
#from pathlib import Path
#from collections import defaultdict
#from collections import Counter
#from PIL import Image, ImageOps
#from PIL import Image, ExifTags 
#import tensorflow as tf
#from PIL import Image 
#import matplotlib.pyplot as plt
#import pandas as pd
#import numpy as np
##data_dir="combined-unknown-pneumonia-and-tuberculosis"
##print(os.listdir(data_dir))
#
#'''checking for any corrupt file by checking the size of the file, 
#if the file has 0 byte size but still there in the folder then it might be 
#a corrupt file.and also checking for how many total files are there in every 
#folder and within each folder, what is the quantity of jpg,png and jpeg files.
#also checking the modes(either RGB or grayscale or anything else) of every image.
#"conclusion from after knowing the mode of every images is that:
#in train/normal : we have, RGB = 63.74% , L = 30.92% and p = 5.95%
#in train/pneumonia: we have, RGB ≈ 7% , L ≈ 93% and p = 0%
#in train/tb: we have, RGB ≈ 31% , L ≈ 61% and p = 7.36%
#
#so from this we can conclude that, this will cause 'Label leakage'
#Label leakage means that model can learn the shortcut that if the img is grayscale
#then predict pneumonia.
#"
#'''
#from pathlib import Path
#
#base_dir = Path(
#    "/data/rishi and dev/rishi's coding stuff/coding stuff/python/python_ptojects/playing with deep learning and new new datasets /combined-unknown-pneumonia-and-tuberculosis"
#)
#dirs = ["NORMAL", "PNEUMONIA", "TUBERCULOSIS"]
#dir_type = ["train", "test", "val"]
#valid_ext = {".jpg", ".png", ".jpeg"}
#for split in dir_type:
#    for label in dirs:
#        current_dir = base_dir / "data" / split / label
#        if not current_dir.exists():
#            print(f"[SKIP] {current_dir} does not exist")
#            continue
#        image_paths = [
#            p for p in current_dir.iterdir()
#            if p.is_file() and p.suffix.lower() in valid_ext
#        ]
#        print(f"\nDirectory: {split}/{label}")
#        print("Total files:", len(image_paths))
#        count = {
#            ".jpeg": 0,
#            ".png": 0,
#            ".jpg": 0
#        }
#        for p in image_paths:
#            count[p.suffix.lower()] += 1
#        print(" .jpeg:", count[".jpeg"])
#        print(" .png :", count[".png"])
#        print(" .jpg :", count[".jpg"])
#
#        bad_files = 0
#        zero_byte_files = 0
#
#        for img_path in current_dir.iterdir():
#            if not img_path.is_file():
#                continue
#            if img_path.suffix.lower() not in valid_ext:
#                continue
#
#            if not img_path.exists():
#                bad_files += 1
#            elif img_path.stat().st_size == 0:
#                zero_byte_files += 1
#
#        print(
#            f"{split}/{label} → "
#            f"bad_files={bad_files}, "
#            f"zero_byte_files={zero_byte_files}"
#        )
#        file_type = {
#            "1": 0,
#            "L": 0,
#            "P": 0,
#            "RGB": 0,
#            "RGBA": 0
#        }
#        for p in image_paths:
#            try:
#                with Image.open(p) as img:
#                    mode = img.mode
#                    if mode in file_type:
#                        file_type[mode] += 1
#                    else:
#                        print(f"[UNKNOWN MODE] {mode} in {p}")
#            except Exception as e:
#                print(f"[BROKEN IMAGE] {p} : {e}")
#        print("\nImage mode distribution:")
#        for mode, count in file_type.items():
#            print(f"{mode}: {count}")
           
#from pathlib import Path
#from PIL import Image 
#from collections import defaultdict
#from PIL import Image, ExifTags
#
## Find Orientation tag ONCE
#ORIENTATION_TAG = None
#for k, v in ExifTags.TAGS.items():
#    if v == "Orientation":
#        ORIENTATION_TAG = k
#        break
#
## Helper: get orientation value
#def get_exif_orientation(path):
#    try:
#        with Image.open(path) as img_ori:
#            exif = img_ori.getexif()
#            if not exif:
#                return None
#            return exif.get(ORIENTATION_TAG, None)
#    except Exception:
#        return None
#    
## Dataset setup
#base_dir = Path(
#    "/data/rishi and dev/rishi's coding stuff/coding stuff/python/python_ptojects/playing with deep learning and new new datasets /combined-unknown-pneumonia-and-tuberculosis"
#)
#dirs = ["NORMAL", "PNEUMONIA", "TUBERCULOSIS"]
#dir_type = ["train", "test", "val"]
#valid_ext = {".jpg", ".png", ".jpeg"}
#sizes = defaultdict(lambda: {"widths": [], "heights": []})
#exif_stats={}
#
#for split in dir_type:
#    for label in dirs:
#        current_dir = base_dir / "data" / split / label
#        if not current_dir.exists():
#            print(f"[SKIP] {current_dir} does not exist")
#            continue
#        image_paths = [
#            p for p in current_dir.iterdir()
#            if p.is_file() and p.suffix.lower() in valid_ext
#        ]
#        key = f"{split}/{label}"
#        try:
#            for img_size in image_paths:
#                with Image.open(img_size) as img:
#                    w,h=img.size
#                    sizes[key]["widths"].append(w)
#                    sizes[key]["heights"].append(h)
#        except Exception as e:
#            print(f"[SKIP IMAGE] {img_size} {e}")
#    
#        total=0
#        with_orientation=0
#        rotation=0
#        for img_ in image_paths:
#            total += 1
#            orientation=get_exif_orientation(img_)
#            if orientation is not None:
#                with_orientation +=1
#                if orientation in{3,6,8}:
#                    rotation +=1
#        exif_stats[f"{split}/{label}"] = {
#                "total": total,
#                "with_orientation":with_orientation,
#                "rotated" : rotation
#        }
#        
##checking how many different sizes does we have and what's their count for whole dataset:
#print("\nCount of image resolutions for the whole dataset: ")
#size_counter=Counter()
#for key,data in sizes.items():
#    for w,h in zip(data["widths"],data["heights"]):
#        size_counter[(w,h)] +=1
#for(w,h),cnt in size_counter.most_common(25):
#    print(f"{w} x {h} : {cnt}")
#
##checking how many different sizes does we have and what's their count for individual folders:
#size_tables={}
#for key,data in sizes.items():
#    c=Counter()
#    for w,h in zip(data["widths"],data["heights"]):
#        c[(w,h)] +=1
#    size_tables[key]=c
#
#for folder,counter in size_tables.items():
#    print(f"\nFolder: {folder}")
#    print(f"-"*30)
#    for (w,h), cnt in counter.most_common(10):
#        print(f"{w} x {h} : {cnt}")
#
#print("\nEXIF ORIENTATION REPORT")
#print("-" * 50)
#for k, v in exif_stats.items():
#    print(
#        f"{k:20} | "
#        f"total={v['total']:4} | "
#        f"orientation_tag={v['with_orientation']:4} | "
#        f"rotated={v['rotated']:4}"
#    )
##print("\nFinal summary:")
##for key, data in sizes.items():
##    print(key, "", len(data["widths"]))
##
#
#
##all_widths = []
##all_heights = []
##
##for data in sizes.values():
##    all_widths.extend(data["widths"])
##    all_heights.extend(data["heights"])
##
##plt.figure(figsize=(6, 6))
##plt.hexbin(all_widths, all_heights, gridsize=50, bins="log")
##plt.colorbar(label="log(count)")
##plt.xlabel("Width (pixels)")
##plt.ylabel("Height (pixels)")
##plt.title("Overall Image Size Distribution")
##plt.savefig("size_distribution.png", dpi=200,bbox_inches="tight")
##plt.close()
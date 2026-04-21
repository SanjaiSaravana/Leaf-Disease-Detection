import os
import json
import nbformat

def create_notebook():
    nb = nbformat.v4.new_notebook()

    # Introduction Markdown
    nb.cells.append(nbformat.v4.new_markdown_cell("""# Hybrid Model for Tomato Disease Detection (CNN + YOLOv8s)
This notebook builds a high-accuracy hybrid pipeline (Target: 99.5+%) using an EfficientNetV2S CNN and a YOLOv8s object detector.

**GPU Memory Fix for GTX 1650 (4GB VRAM):**
If you have experienced kernel failures at the start of an epoch, this is typically due to Out-Of-Memory (OOM) errors. We have added memory limiters and smaller batch sizes specifically for both TensorFlow and PyTorch to resolve this.
"""))

    # Cell 1: Environment Setup
    nb.cells.append(nbformat.v4.new_code_cell("""# Import dependencies
import os
import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt

# Ultralytics for YOLOv8
# !pip install ultralytics
from ultralytics import YOLO

import torch
"""))

    # Cell 2: GPU Memory Fix specifically for GTX 1650
    nb.cells.append(nbformat.v4.new_code_cell("""# ==========================================
# CRITICAL: GPU MEMORY GROWTH (ANTI-CRASH)
# ==========================================
# This prevents TensorFlow from allocating all 4GB of GTX 1650 VRAM at once, which causes kernel crashes.
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ Memory growth set for {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(e)
        
# For PyTorch (YOLOv8)
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print("✅ PyTorch CUDA cache cleared")
"""))

    # Cell 3: Data Preparation for CNN
    nb.cells.append(nbformat.v4.new_markdown_cell("## 1. CNN Classification (EfficientNetV2S) \nWe will extract the image-level classes by parsing the YOLO `.txt` label files."))
    
    nb.cells.append(nbformat.v4.new_code_cell("""# Define dataset paths
dataset_path = "."  # We are already in the 'cnn+yolo' folder
train_images = os.path.join(dataset_path, "train", "images")
train_labels = os.path.join(dataset_path, "train", "labels")
valid_images = os.path.join(dataset_path, "valid", "images")
valid_labels = os.path.join(dataset_path, "valid", "labels")
test_images = os.path.join(dataset_path, "test", "images")
test_labels = os.path.join(dataset_path, "test", "labels")

# Class names based on data.yaml
class_names = ['Corn Common Rust', 'Peach_Bacterial_Spot', 'Strawberry leaf scorch', 'Tomato Late Blight']

def create_dataframe_from_yolo(images_dir, labels_dir):
    data = []
    for txt_file in os.listdir(labels_dir):
        if not txt_file.endswith('.txt'): continue
        
        # Read the first label in the txt file
        with open(os.path.join(labels_dir, txt_file), 'r') as f:
            lines = f.readlines()
            if not lines: continue
            
            # Extract class index (first integer in the row)
            class_idx = int(lines[0].strip().split(' ')[0])
            class_name = class_names[class_idx]
            
            # The image name corresponds to the txt name
            img_name = txt_file.replace('.txt', '.jpg')
            if os.path.exists(os.path.join(images_dir, img_name)):
                data.append({'filename': img_name, 'class': class_name})
                
    return pd.DataFrame(data)

df_train = create_dataframe_from_yolo(train_images, train_labels)
df_valid = create_dataframe_from_yolo(valid_images, valid_labels)
df_test = create_dataframe_from_yolo(test_images, test_labels)

print(f"Training samples: {len(df_train)}")
print(f"Validation samples: {len(df_valid)}")
"""))

    # Cell 4: ImageDataGenerators
    nb.cells.append(nbformat.v4.new_code_cell("""# Use small batch sizes (8 or 16) to prevent GTX 1650 Kernel Crashing
BATCH_SIZE = 16 
IMG_SIZE = (224, 224)

# Data augmentation for robust 99.5+% accuracy
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

valid_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=df_train,
    directory=train_images,
    x_col="filename",
    y_col="class",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

valid_generator = valid_datagen.flow_from_dataframe(
    dataframe=df_valid,
    directory=valid_images,
    x_col="filename",
    y_col="class",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)
"""))

    # Cell 5: CNN Model Build
    nb.cells.append(nbformat.v4.new_code_cell("""# Build the CNN using EfficientNetV2S
base_model = EfficientNetV2S(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model for initial warm-up
base_model.trainable = False 

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(len(class_names), activation='softmax')(x)

cnn_model = Model(inputs=base_model.input, outputs=predictions)

cnn_model.compile(optimizer=Adam(learning_rate=1e-3), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])

cnn_model.summary()
"""))

    # Cell 6: CNN Train Phase 1
    nb.cells.append(nbformat.v4.new_code_cell("""# Phase 1: Train the top layers
callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1),
    ModelCheckpoint('best_cnn_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
]

print("Starting Phase 1 Training (Warm-up)...")
history1 = cnn_model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=5,
    callbacks=callbacks
)
"""))

    # Cell 7: CNN Train Phase 2 (Fine-tuning)
    nb.cells.append(nbformat.v4.new_code_cell("""# Phase 2: Fine-Tuning the entire model to push for 99.5+%
base_model.trainable = True

# Explicitly garbage collect to free memory before fine-tuning
import gc
gc.collect()

# We use a very small learning rate for fine-tuning
cnn_model.compile(optimizer=Adam(learning_rate=1e-5), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])

print("Starting Phase 2 Training (Fine-tuning)...")
history2 = cnn_model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=15, 
    callbacks=callbacks
)
"""))

    # Cell 8: YOLOv8 Training
    nb.cells.append(nbformat.v4.new_markdown_cell("## 2. YOLOv8s Object Detection"))
    
    nb.cells.append(nbformat.v4.new_code_cell("""# Clear session memory before loading YOLO
tf.keras.backend.clear_session()
import gc
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Load YOLOv8s pre-trained model
yolo_model = YOLO("yolov8s.pt")  

# Train YOLOv8s natively on the data.yaml
# batch=16 fits on 4GB VRAM. If kernel crashes here, change batch=8. Enable AMP (mixed precision).
results = yolo_model.train(
    data="data.yaml", 
    epochs=25, 
    imgsz=640, 
    batch=16, 
    name="tomato_yolo",
    amp=True # Mixed precision helps with GTX 1650 memory
)
"""))

    # Cell 9: CNN+YOLO Inference Pipeline
    nb.cells.append(nbformat.v4.new_markdown_cell("## 3. Hybrid Inference Pipeline \nPredicting bounding boxes via YOLO, and scoring overall disease via EfficientNetV2S."))

    nb.cells.append(nbformat.v4.new_code_cell("""# Load the chosen test image
import random
import matplotlib.patches as patches

# Helper to predict
def hybrid_predict(img_path):
    # 1. CNN Classification
    img_cnn = tf.keras.preprocessing.image.load_img(img_path, target_size=IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img_cnn) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    cnn_preds = cnn_model.predict(img_array)[0]
    best_idx = np.argmax(cnn_preds)
    cnn_conf = cnn_preds[best_idx]
    cnn_class = class_names[best_idx]
    
    # 2. YOLOv8 Detection
    det_results = yolo_model.predict(img_path, conf=0.25, verbose=False)
    
    # Visualization
    img_cv = cv2.imread(img_path)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    
    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.imshow(img_cv)
    ax.set_title(f"CNN: {cnn_class} ({cnn_conf:.2%})")
    
    # Draw YOLO boxes
    for r in det_results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            cls_idx = int(box.cls[0])
            label = f"{class_names[cls_idx]} {conf:.2f}"
            
            w, h = x2 - x1, y2 - y1
            rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor='lime', facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1 - 10, label, color='lime', fontsize=12, weight='bold')

    plt.axis('off')
    plt.show()

# Test on a random test image
sample_images = [img for img in os.listdir(test_images) if img.endswith('.jpg')]
if sample_images:
    random_img = random.choice(sample_images)
    hybrid_predict(os.path.join(test_images, random_img))
else:
    print("No test images found.")
"""))

    with open('hybrid_model.ipynb', 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

if __name__ == "__main__":
    create_notebook()

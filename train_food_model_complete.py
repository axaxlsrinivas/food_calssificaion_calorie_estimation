"""
Complete Food Recognition Model Training Script
Run this entire script in Google Colab with GPU enabled

Instructions:
1. Open Google Colab: https://colab.research.google.com/
2. Create new notebook
3. Runtime -> Change runtime type -> GPU (T4 GPU)
4. Copy this ENTIRE file into ONE cell
5. Run the cell (takes 1.5-4 hours)
6. Download the files when complete
"""

# ============================================================
# INSTALL AND IMPORT
# ============================================================
print("Installing packages...")
import os
os.system('pip install -q tensorflow pillow matplotlib')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import json

print("\n" + "="*60)
print("FOOD RECOGNITION MODEL TRAINING")
print("="*60)
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")
print("="*60)

# ============================================================
# STEP 1: DOWNLOAD FOOD-101 DATASET
# ============================================================
print("\n" + "="*60)
print("STEP 1: DOWNLOADING DATASET")
print("="*60)

if not os.path.exists("food-101"):
    print("Downloading Food-101 dataset (~5GB, 10-20 minutes)...")
    os.system("wget -q http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz")
    print("Extracting dataset...")
    os.system("tar -xzf food-101.tar.gz")
    print("[OK] Dataset ready!")
else:
    print("[OK] Dataset already exists")

# ============================================================
# STEP 2: CONFIGURE TRAINING
# ============================================================
print("\n" + "="*60)
print("STEP 2: CONFIGURATION")
print("="*60)

# Training configuration
USE_SUBSET = True  # Set to False for all 101 classes
NUM_CLASSES = 25   # Only used if USE_SUBSET=True
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
EPOCHS_PHASE1 = 10
EPOCHS_PHASE2 = 10

print(f"Training mode: {'SUBSET (25 foods)' if USE_SUBSET else 'FULL (101 foods)'}")
print(f"Image size: {IMG_HEIGHT}x{IMG_WIDTH}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Phase 1 epochs: {EPOCHS_PHASE1}")
print(f"Phase 2 epochs: {EPOCHS_PHASE2}")

# ============================================================
# STEP 3: PREPARE DATASET
# ============================================================
print("\n" + "="*60)
print("STEP 3: PREPARING DATASET")
print("="*60)

images_dir = "food-101/images"
all_classes = sorted([d for d in os.listdir(images_dir) 
                     if os.path.isdir(os.path.join(images_dir, d))])

print(f"Total available classes: {len(all_classes)}")

if USE_SUBSET:
    # Use popular food classes
    popular_classes = [
        'apple_pie', 'pizza', 'hamburger', 'sushi', 'steak',
        'ice_cream', 'french_fries', 'chocolate_cake', 'spaghetti_carbonara',
        'chicken_curry', 'fried_rice', 'ramen', 'grilled_salmon', 'pancakes',
        'donuts', 'hot_dog', 'tacos', 'lasagna', 'caesar_salad', 'pad_thai',
        'tiramisu', 'baklava', 'beignets', 'churros', 'guacamole'
    ]
    class_names = [c for c in popular_classes if c in all_classes][:NUM_CLASSES]
    print(f"Using {len(class_names)} popular foods")
else:
    class_names = all_classes
    print(f"Using all {len(class_names)} classes")

print(f"Classes: {', '.join(class_names[:5])}...")

# Create datasets
print("\nCreating training and validation datasets...")

train_ds = keras.utils.image_dataset_from_directory(
    images_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    labels='inferred',
    label_mode='categorical',
    class_names=class_names
)

val_ds = keras.utils.image_dataset_from_directory(
    images_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    labels='inferred',
    label_mode='categorical',
    class_names=class_names
)

# Optimize performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

print(f"[OK] Training batches: {len(train_ds)}")
print(f"[OK] Validation batches: {len(val_ds)}")

# ============================================================
# STEP 4: BUILD MODEL
# ============================================================
print("\n" + "="*60)
print("STEP 4: BUILDING MODEL")
print("="*60)

# Data augmentation
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
], name="data_augmentation")

# Load pre-trained MobileNetV2
print("Loading MobileNetV2 with ImageNet weights...")
base_model = keras.applications.MobileNetV2(
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze base model
base_model.trainable = False

# Build full model
inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
x = data_augmentation(inputs)
x = keras.applications.mobilenet_v2.preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(len(class_names), activation='softmax')(x)

model = keras.Model(inputs, outputs, name="food_recognition_model")

# Compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy')]
)

print("\n")
model.summary()
print(f"\n[OK] Model created with {len(class_names)} classes")

# ============================================================
# STEP 5: TRAIN PHASE 1 (Classification Head)
# ============================================================
print("\n" + "="*60)
print("STEP 5: TRAINING PHASE 1 - Classification Head")
print("="*60)
print("Base model: FROZEN")
print(f"Training for {EPOCHS_PHASE1} epochs...")

checkpoint1 = keras.callbacks.ModelCheckpoint(
    'food_model_phase1.h5',
    save_best_only=True,
    monitor='val_accuracy',
    mode='max',
    verbose=1
)

early_stop1 = keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=3,
    restore_best_weights=True,
    verbose=1
)

reduce_lr1 = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=2,
    min_lr=1e-7,
    verbose=1
)

history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_PHASE1,
    callbacks=[checkpoint1, early_stop1, reduce_lr1],
    verbose=1
)

best_acc1 = max(history1.history['val_accuracy'])
print(f"\n[OK] Phase 1 Complete!")
print(f"Best Validation Accuracy: {best_acc1:.4f} ({best_acc1*100:.2f}%)")

# ============================================================
# STEP 6: TRAIN PHASE 2 (Fine-tuning)
# ============================================================
print("\n" + "="*60)
print("STEP 6: TRAINING PHASE 2 - Fine-tuning")
print("="*60)

# Unfreeze base model
base_model.trainable = True

# Freeze first 100 layers
for layer in base_model.layers[:100]:
    layer.trainable = False

trainable_layers = sum([1 for layer in base_model.layers if layer.trainable])
print(f"Base model: {len(base_model.layers)} total layers")
print(f"Trainable layers: {trainable_layers}")

# Recompile with lower learning rate
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy')]
)

print(f"\nTraining for {EPOCHS_PHASE2} more epochs...")

checkpoint2 = keras.callbacks.ModelCheckpoint(
    'food_model_final.h5',
    save_best_only=True,
    monitor='val_accuracy',
    mode='max',
    verbose=1
)

early_stop2 = keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

reduce_lr2 = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=2,
    min_lr=1e-8,
    verbose=1
)

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_PHASE2,
    callbacks=[checkpoint2, early_stop2, reduce_lr2],
    verbose=1
)

best_acc2 = max(history2.history['val_accuracy'])
print(f"\n[OK] Phase 2 Complete!")
print(f"Best Validation Accuracy: {best_acc2:.4f} ({best_acc2*100:.2f}%)")

# ============================================================
# STEP 7: EVALUATE MODEL
# ============================================================
print("\n" + "="*60)
print("STEP 7: FINAL EVALUATION")
print("="*60)

results = model.evaluate(val_ds, verbose=1)

print("\n" + "="*60)
print("FINAL RESULTS")
print("="*60)
print(f"Validation Loss: {results[0]:.4f}")
print(f"Validation Accuracy: {results[1]:.4f} ({results[1]*100:.2f}%)")
print(f"Top-5 Accuracy: {results[2]:.4f} ({results[2]*100:.2f}%)")
print("="*60)

# ============================================================
# STEP 8: PLOT TRAINING HISTORY
# ============================================================
print("\n" + "="*60)
print("STEP 8: GENERATING PLOTS")
print("="*60)

# Combine histories
acc = history1.history['accuracy'] + history2.history['accuracy']
val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
loss = history1.history['loss'] + history2.history['loss']
val_loss = history1.history['val_loss'] + history2.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(14, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.axvline(x=len(history1.history['accuracy']), color='r', linestyle='--', label='Fine-tuning starts')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.axvline(x=len(history1.history['loss']), color='r', linestyle='--', label='Fine-tuning starts')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.tight_layout()
plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
print("[OK] Saved: training_history.png")
plt.show()

# ============================================================
# STEP 9: SAVE MODEL AND CLASS NAMES
# ============================================================
print("\n" + "="*60)
print("STEP 9: SAVING MODEL FILES")
print("="*60)

# Save final model in H5 format
print("Saving model in H5 format...")
model.save('food_model_trained.h5', save_format='h5')
print("[OK] Saved: food_model_trained.h5")

# Save class names
with open('class_names.json', 'w') as f:
    json.dump(class_names, f, indent=2)
print("[OK] Saved: class_names.json")

# ============================================================
# STEP 10: TEST PREDICTIONS
# ============================================================
print("\n" + "="*60)
print("STEP 10: TESTING PREDICTIONS")
print("="*60)

# Get a batch
images_batch, labels_batch = next(iter(val_ds))

# Make predictions
predictions = model.predict(images_batch[:9])

# Plot results
plt.figure(figsize=(15, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(images_batch[i].numpy().astype("uint8"))
    
    predicted_class = class_names[np.argmax(predictions[i])]
    true_class = class_names[np.argmax(labels_batch[i])]
    confidence = np.max(predictions[i]) * 100
    
    color = 'green' if predicted_class == true_class else 'red'
    plt.title(f"Pred: {predicted_class}\n({confidence:.1f}%)\nTrue: {true_class}", 
             color=color, fontsize=9)
    plt.axis('off')

plt.tight_layout()
plt.savefig('sample_predictions.png', dpi=150, bbox_inches='tight')
print("[OK] Saved: sample_predictions.png")
plt.show()

# ============================================================
# STEP 11: DOWNLOAD FILES
# ============================================================
print("\n" + "="*60)
print("STEP 11: DOWNLOADING FILES")
print("="*60)

try:
    from google.colab import files
    
    print("Downloading food_model_trained.h5...")
    files.download('food_model_trained.h5')
    
    print("Downloading class_names.json...")
    files.download('class_names.json')
    
    print("Downloading training_history.png...")
    files.download('training_history.png')
    
    print("Downloading sample_predictions.png...")
    files.download('sample_predictions.png')
    
    print("\n[OK] All files downloaded!")
    
except ImportError:
    print("\nNot running in Colab - files saved locally")

# ============================================================
# TRAINING COMPLETE!
# ============================================================
print("\n" + "="*60)
print("[OK] TRAINING COMPLETE!")
print("="*60)
print(f"Model accuracy: {results[1]*100:.2f}%")
print(f"Top-5 accuracy: {results[2]*100:.2f}%")
print(f"Classes trained: {len(class_names)}")
print("\nFiles created:")
print("  - food_model_trained.h5 (trained model)")
print("  - class_names.json (food categories)")
print("  - training_history.png (training curves)")
print("  - sample_predictions.png (test results)")
print("\nNext steps:")
print("1. Download the files from Colab")
print("2. Copy food_model_trained.h5 to models/food_model.h5")
print("3. Copy class_names.json to models/class_names.json")
print("4. Restart your API server")
print("5. Test with real images!")
print("="*60)

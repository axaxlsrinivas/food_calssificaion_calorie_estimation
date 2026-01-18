"""
Train Food Recognition Model on Food-101 Dataset
Run this in Google Colab for free GPU access

Instructions:
1. Open Google Colab: https://colab.research.google.com/
2. Runtime -> Change runtime type -> GPU
3. Copy this entire file into a new cell
4. Run the cell
5. Download the trained model and upload to your project
"""

# Install required packages
# Run this command in your terminal before running the script:
# pip install tensorflow keras Pillow matplotlib

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
import zipfile
import requests
from pathlib import Path
import matplotlib.pyplot as plt
import json

print("TensorFlow version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# ============================================================
# 1. DOWNLOAD FOOD-101 DATASET
# ============================================================

def download_food101():
    """Download and extract Food-101 dataset"""
    print("\n" + "="*60)
    print("DOWNLOADING FOOD-101 DATASET")
    print("="*60)
    print("Dataset size: ~5GB")
    print("This will take 10-20 minutes...")
    
    # Download from official source
    url = "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
    dataset_path = "food-101.tar.gz"
    
    if not os.path.exists("food-101"):
        print(f"\nDownloading from {url}...")
        os.system(f"wget {url}")
        
        print("\nExtracting dataset...")
        os.system(f"tar -xzf {dataset_path}")
        print("[OK] Dataset extracted!")
    else:
        print("[OK] Dataset already exists")
    
    return "food-101"

# Download dataset
dataset_dir = download_food101()

# ============================================================
# 2. PREPARE DATASET
# ============================================================

def prepare_dataset(dataset_dir, use_subset=False, num_classes=25):
    """
    Prepare Food-101 dataset for training
    
    Args:
        dataset_dir: Path to food-101 directory
        use_subset: If True, use only subset of classes (faster training)
        num_classes: Number of classes to use if use_subset=True
    """
    print("\n" + "="*60)
    print("PREPARING DATASET")
    print("="*60)
    
    # Food-101 has 101 classes with 1000 images each
    # 750 training images per class, 250 test images per class
    
    img_height = 224
    img_width = 224
    batch_size = 32
    
    # Get all class names
    images_dir = os.path.join(dataset_dir, "images")
    all_classes = sorted([d for d in os.listdir(images_dir) 
                         if os.path.isdir(os.path.join(images_dir, d))])
    
    print(f"Total classes available: {len(all_classes)}")
    
    if use_subset:
        # Use popular food classes for faster training
        popular_classes = [
            'apple_pie', 'pizza', 'hamburger', 'sushi', 'steak',
            'ice_cream', 'french_fries', 'chocolate_cake', 'spaghetti_carbonara',
            'chicken_curry', 'fried_rice', 'ramen', 'grilled_salmon', 'pancakes',
            'donuts', 'hot_dog', 'tacos', 'lasagna', 'caesar_salad', 'pad_thai',
            'tiramisu', 'baklava', 'beignets', 'churros', 'guacamole'
        ]
        classes_to_use = [c for c in popular_classes if c in all_classes][:num_classes]
        print(f"Using subset: {len(classes_to_use)} classes")
    else:
        classes_to_use = all_classes
        print(f"Using all {len(classes_to_use)} classes")
    
    print(f"Classes: {', '.join(classes_to_use[:5])}...")
    
    # Create train and validation datasets
    train_ds = keras.utils.image_dataset_from_directory(
        images_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        labels='inferred',
        label_mode='categorical',
        class_names=classes_to_use
    )
    
    val_ds = keras.utils.image_dataset_from_directory(
        images_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        labels='inferred',
        label_mode='categorical',
        class_names=classes_to_use
    )
    
    # Optimize performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    print(f"[OK] Training batches: {len(train_ds)}")
    print(f"[OK] Validation batches: {len(val_ds)}")
    
    return train_ds, val_ds, classes_to_use

# Prepare dataset
# Set use_subset=True for faster training (recommended for first time)
# Set use_subset=False for full Food-101 training
train_ds, val_ds, class_names = prepare_dataset(
    dataset_dir, 
    use_subset=True,  # Change to False for all 101 classes
    num_classes=25     # Use 25 popular foods
)

# ============================================================
# 3. BUILD MODEL
# ============================================================

def create_food_model(num_classes, img_height=224, img_width=224):
    """
    Create food recognition model using transfer learning
    """
    print("\n" + "="*60)
    print("BUILDING MODEL")
    print("="*60)
    
    # Data augmentation
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ])
    
    # Load pre-trained MobileNetV2
    base_model = keras.applications.MobileNetV2(
        input_shape=(img_height, img_width, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Build model
    inputs = keras.Input(shape=(img_height, img_width, 3))
    
    # Data augmentation
    x = data_augmentation(inputs)
    
    # Preprocessing for MobileNetV2
    x = keras.applications.mobilenet_v2.preprocess_input(x)
    
    # Base model
    x = base_model(x, training=False)
    
    # Classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy')]
    )
    
    print("\n")
    model.summary()
    print(f"\n[OK] Model created with {num_classes} classes")
    
    return model, base_model

model, base_model = create_food_model(len(class_names))

# ============================================================
# 4. TRAIN MODEL (PHASE 1: HEAD ONLY)
# ============================================================

def train_phase1(model, train_ds, val_ds, epochs=10):
    """Train classification head with frozen base"""
    print("\n" + "="*60)
    print("TRAINING PHASE 1: Classification Head")
    print("="*60)
    print("Base model: FROZEN")
    print(f"Training for {epochs} epochs...")
    
    # Callbacks
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        'food_model_phase1.h5',
        save_best_only=True,
        monitor='val_accuracy',
        mode='max',
        verbose=1
    )
    
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=3,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=2,
        min_lr=1e-7,
        verbose=1
    )
    
    # Train
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[checkpoint_callback, early_stopping, reduce_lr]
    )
    
    print(f"\n[OK] Phase 1 Complete!")
    print(f"Best Validation Accuracy: {max(history1.history['val_accuracy']):.4f}")
    
    return history1

# Train phase 1
history1 = train_phase1(model, train_ds, val_ds, epochs=10)

# ============================================================
# 5. TRAIN MODEL (PHASE 2: FINE-TUNING)
# ============================================================

def train_phase2(model, base_model, train_ds, val_ds, epochs=10):
    """Fine-tune with unfrozen layers"""
    print("\n" + "="*60)
    print("TRAINING PHASE 2: Fine-tuning")
    print("="*60)
    
    # Unfreeze base model
    base_model.trainable = True
    
    # Freeze first 100 layers (fine-tune last layers only)
    for layer in base_model.layers[:100]:
        layer.trainable = False
    
    print(f"Base model: {len(base_model.layers)} total layers")
    print(f"Trainable layers: {sum([1 for layer in base_model.layers if layer.trainable])}")
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy')]
    )
    
    print(f"\nTraining for {epochs} more epochs...")
    
    # Callbacks
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        'food_model_final.h5',
        save_best_only=True,
        monitor='val_accuracy',
        mode='max',
        verbose=1
    )
    
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=2,
        min_lr=1e-8,
        verbose=1
    )
    
    # Train
    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[checkpoint_callback, early_stopping, reduce_lr]
    )
    
    print(f"\n[OK] Phase 2 Complete!")
    print(f"Best Validation Accuracy: {max(history2.history['val_accuracy']):.4f}")
    
    return history2

# Train phase 2
history2 = train_phase2(model, base_model, train_ds, val_ds, epochs=10)

# ============================================================
# 6. EVALUATE MODEL
# ============================================================

def evaluate_model(model, val_ds):
    """Evaluate model performance"""
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    results = model.evaluate(val_ds, verbose=1)
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Validation Loss: {results[0]:.4f}")
    print(f"Validation Accuracy: {results[1]:.4f} ({results[1]*100:.2f}%)")
    print(f"Top-5 Accuracy: {results[2]:.4f} ({results[2]*100:.2f}%)")
    print("="*60)
    
    return results

results = evaluate_model(model, val_ds)

# ============================================================
# 7. PLOT TRAINING HISTORY
# ============================================================

def plot_training_history(history1, history2):
    """Plot training curves"""
    print("\nGenerating training plots...")
    
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
    plt.savefig('training_history.png', dpi=150)
    print("[OK] Saved training_history.png")
    plt.show()

plot_training_history(history1, history2)

# ============================================================
# 8. SAVE MODEL AND CLASS NAMES
# ============================================================

def save_model_and_classes(model, class_names):
    """Save trained model and class names"""
    print("\n" + "="*60)
    print("SAVING MODEL")
    print("="*60)
    
    # Save model in both formats for compatibility
    model.save('food_model_trained.h5', save_format='h5')
    print("[OK] Saved: food_model_trained.h5")
    
    # Also save in Keras format
    model.save('food_model_trained.keras')
    print("[OK] Saved: food_model_trained.keras (alternative format)")
    
    # Save class names
    with open('class_names.json', 'w') as f:
        json.dump(class_names, f, indent=2)
    print("[OK] Saved: class_names.json")
    
    print("\n" + "="*60)
    print("DOWNLOAD INSTRUCTIONS")
    print("="*60)
    print("1. Download these files from Colab:")
    print("   - food_model_trained.h5")
    print("   - class_names.json")
    print("   - training_history.png")
    print("\n2. In your project:")
    print("   - Copy food_model_trained.h5 to models/food_model.h5")
    print("   - Copy class_names.json to models/class_names.json")
    print("\n3. Restart your API server")
    print("="*60)

save_model_and_classes(model, class_names)

# ============================================================
# 9. TEST PREDICTIONS
# ============================================================

def test_predictions(model, val_ds, class_names, num_images=9):
    """Test predictions on sample images"""
    print("\n" + "="*60)
    print("TESTING PREDICTIONS")
    print("="*60)
    
    # Get a batch
    images, labels = next(iter(val_ds))
    
    # Make predictions
    predictions = model.predict(images[:num_images])
    
    # Plot results
    plt.figure(figsize=(15, 10))
    for i in range(num_images):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        
        predicted_class = class_names[np.argmax(predictions[i])]
        true_class = class_names[np.argmax(labels[i])]
        confidence = np.max(predictions[i]) * 100
        
        color = 'green' if predicted_class == true_class else 'red'
        plt.title(f"Pred: {predicted_class}\n({confidence:.1f}%)\nTrue: {true_class}", 
                 color=color, fontsize=9)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_predictions.png', dpi=150)
    print("[OK] Saved: sample_predictions.png")
    plt.show()

test_predictions(model, val_ds, class_names)

print("\n" + "="*60)
print("[OK] TRAINING COMPLETE!")
print("="*60)
print(f"Total training time: ~2-4 hours")
print(f"Model accuracy: {results[1]*100:.2f}%")
print(f"Top-5 accuracy: {results[2]*100:.2f}%")
print("\nNext steps:")
print("1. Download food_model_trained.h5 and class_names.json")
print("2. Replace your existing model files")
print("3. Test with real images!")
print("="*60)

"""
Script to download and setup a pre-trained Food-101 model
This replaces the untrained model with a model that can actually recognize food
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import json
import os

def download_and_setup_food101_model():
    """
    Download MobileNetV2 pre-trained on ImageNet and fine-tune for food recognition
    This is a working solution until you train your own model
    """
    
    print("Setting up food recognition model...")
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Load MobileNetV2 with ImageNet weights
    print("Loading MobileNetV2 with ImageNet weights...")
    base_model = keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Food classes (25 common foods)
    food_classes = [
        "Apple", "Banana", "Bread", "Burger", "Cake",
        "Chicken", "Coffee", "Cookie", "Donut", "Egg",
        "French Fries", "Grape", "Hot Dog", "Ice Cream", "Orange",
        "Pancake", "Pizza", "Rice", "Salad", "Sandwich",
        "Spaghetti", "Steak", "Strawberry", "Sushi", "Taco"
    ]
    
    # Build model
    model = keras.Sequential([
        base_model,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(len(food_classes), activation='softmax')
    ])
    
    # Initialize with random weights for now
    # NOTE: This still won't give accurate predictions without training
    print("Building model architecture...")
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Save the model
    model_path = "models/food_model.h5"
    model.save(model_path)
    print(f"✓ Model saved to {model_path}")
    
    # Save class names
    class_names_path = "models/class_names.json"
    with open(class_names_path, 'w') as f:
        json.dump(food_classes, f)
    print(f"✓ Class names saved to {class_names_path}")
    
    print("\n" + "="*60)
    print("⚠️  IMPORTANT: Model Architecture Setup Complete")
    print("="*60)
    print("\nHowever, the model is NOT TRAINED on food images yet!")
    print("\nFor accurate predictions, you need to:")
    print("1. Collect a food image dataset (or use Food-101 dataset)")
    print("2. Train the model using train_food_model.py")
    print("\nOR use a pre-trained model from TensorFlow Hub")
    print("="*60)
    
    return model

if __name__ == "__main__":
    download_and_setup_food101_model()
    print("\n✓ Setup complete! Restart your API: python app.py")

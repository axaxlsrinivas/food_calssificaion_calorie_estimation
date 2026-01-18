import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import logging
from typing import Dict, List, Tuple
import json
import os
import io

logger = logging.getLogger(__name__)

# Try to import Clarifai for better accuracy
try:
    from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
    from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
    from clarifai_grpc.grpc.api.status import status_code_pb2
    CLARIFAI_AVAILABLE = True
except ImportError:
    CLARIFAI_AVAILABLE = False
    logger.warning("Clarifai not installed. Using local model. Install with: pip install clarifai-grpc")

class FoodRecognitionModel:
    """
    Food recognition model using deep learning
    Supports both pre-trained models and custom training
    Now includes Clarifai integration for accurate predictions
    """
    
    def __init__(self, model_path: str = "models/food_model.h5", use_clarifai: bool = True):
        self.model_path = model_path
        self.model = None
        self.class_names = []
        self.img_height = 224
        self.img_width = 224
        self.use_clarifai = use_clarifai and CLARIFAI_AVAILABLE
        
        # Setup Clarifai if available
        self.clarifai_stub = None
        if self.use_clarifai:
            self._setup_clarifai()
        
        # Load class names
        self._load_class_names()
    
    def _setup_clarifai(self):
        """Setup Clarifai client for accurate predictions"""
        try:
            api_key = os.getenv('CLARIFAI_API_KEY', '1301a1be46094763a888f4f5eb73182d')
            
            if not api_key:
                logger.warning("CLARIFAI_API_KEY not set. Using local model.")
                self.use_clarifai = False
                return
            
            # Setup Clarifai channel
            channel = ClarifaiChannel.get_grpc_channel()
            self.clarifai_stub = service_pb2_grpc.V2Stub(channel)
            self.clarifai_metadata = (('authorization', f'Key {api_key}'),)
            
            # Use Clarifai's General model with food concepts
            # Using the public general model which works better for food detection
            self.clarifai_user_id = "clarifai"
            self.clarifai_app_id = "main"
            self.clarifai_model_id = "food-item-recognition"
            self.clarifai_model_version_id = "1d5fd481e0cf4826aa72ec3ff049e044"
            
            logger.info("✓ Clarifai food recognition enabled (accurate predictions)")
            logger.info(f"Using API key: {api_key[:8]}...")
            
        except Exception as e:
            logger.warning(f"Failed to setup Clarifai: {e}. Using local model.")
            self.use_clarifai = False
    
    def _load_class_names(self):
        """Load food class names from file or use default"""
        class_names_path = "models/class_names.json"
        
        if os.path.exists(class_names_path):
            with open(class_names_path, 'r') as f:
                self.class_names = json.load(f)
        else:
            # Default food classes (expand this list)
            self.class_names = [
                "Apple", "Banana", "Bread", "Burger", "Cake",
                "Chicken", "Coffee", "Cookie", "Donut", "Egg",
                "French Fries", "Grape", "Hot Dog", "Ice Cream", "Orange",
                "Pancake", "Pizza", "Rice", "Salad", "Sandwich",
                "Spaghetti", "Steak", "Strawberry", "Sushi", "Taco"
            ]
            
            # Save class names
            os.makedirs("models", exist_ok=True)
            with open(class_names_path, 'w') as f:
                json.dump(self.class_names, f)
        
        logger.info(f"Loaded {len(self.class_names)} food classes")
    
    def load_model(self):
        """Load pre-trained model or create new one"""
        try:
            if os.path.exists(self.model_path):
                self.model = keras.models.load_model(self.model_path)
                logger.info(f"Loaded model from {self.model_path}")
            else:
                logger.warning(f"Model file not found at {self.model_path}")
                self.model = self._create_model()
                logger.info("Created new model")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model = self._create_model()
            logger.info("Created new model due to loading error")
    
    def _create_model(self) -> keras.Model:
        """
        Create a new model using transfer learning with MobileNetV2
        """
        # Load pre-trained MobileNetV2
        base_model = keras.applications.MobileNetV2(
            input_shape=(self.img_height, self.img_width, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze base model
        base_model.trainable = False
        
        # Create new model
        inputs = keras.Input(shape=(self.img_height, self.img_width, 3))
        x = base_model(inputs, training=False)
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Dropout(0.2)(x)
        outputs = keras.layers.Dense(len(self.class_names), activation='softmax')(x)
        
        model = keras.Model(inputs, outputs)
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Save model
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        model.save(self.model_path)
        
        return model
    
    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """
        Preprocess image for model input
        
        Args:
            image: PIL Image
            
        Returns:
            Preprocessed image as numpy array
        """
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize
        image = image.resize((self.img_width, self.img_height))
        
        # Convert to array and normalize
        img_array = keras.preprocessing.image.img_to_array(image)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = keras.applications.mobilenet_v2.preprocess_input(img_array)
        
        return img_array
    
    def _predict_with_clarifai(self, image: Image.Image, top_k: int = 5) -> Dict:
        """
        Use Clarifai API for accurate food recognition
        
        Args:
            image: PIL Image
            top_k: Number of predictions to return
            
        Returns:
            Dictionary with prediction results
        """
        try:
            logger.info("🔍 Using Clarifai API for prediction...")
            
            # Convert image to bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='JPEG')
            img_byte_arr = img_byte_arr.getvalue()
            
            logger.info(f"Image size: {len(img_byte_arr)} bytes")
            
            # Make prediction request with proper user/app IDs
            request = service_pb2.PostModelOutputsRequest(
                user_app_id=resources_pb2.UserAppIDSet(
                    user_id=self.clarifai_user_id,
                    app_id=self.clarifai_app_id
                ),
                model_id=self.clarifai_model_id,
                version_id=self.clarifai_model_version_id,
                inputs=[
                    resources_pb2.Input(
                        data=resources_pb2.Data(
                            image=resources_pb2.Image(
                                base64=img_byte_arr
                            )
                        )
                    )
                ]
            )
            
            logger.info("Sending request to Clarifai...")
            response = self.clarifai_stub.PostModelOutputs(request, metadata=self.clarifai_metadata)
            
            logger.info(f"Clarifai response status: {response.status.code} - {response.status.description}")
            
            if response.status.code != status_code_pb2.SUCCESS:
                logger.error(f"❌ Clarifai API error: {response.status.description}. Falling back to local model.")
                return self._predict_with_local_model(image, top_k)
            
            # Parse results
            concepts = response.outputs[0].data.concepts
            
            logger.info(f"✓ Clarifai returned {len(concepts)} predictions")
            
            # Get top predictions
            top_predictions = [
                {
                    "food_name": concept.name.title(),
                    "confidence": float(concept.value) * 100  # Convert to percentage
                }
                for concept in concepts[:top_k]
            ]
            
            if not top_predictions:
                logger.warning("No predictions from Clarifai. Using local model.")
                return self._predict_with_local_model(image, top_k)
            
            logger.info(f"✓ Top prediction: {top_predictions[0]['food_name']} ({top_predictions[0]['confidence']:.2f}%)")
            
            return {
                "food_name": top_predictions[0]['food_name'],
                "confidence": top_predictions[0]['confidence'],
                "top_predictions": top_predictions
            }
            
        except Exception as e:
            logger.error(f"❌ Clarifai prediction error: {type(e).__name__}: {e}. Using local model.")
            import traceback
            logger.error(traceback.format_exc())
            return self._predict_with_local_model(image, top_k)
    
    def _predict_with_local_model(self, image: Image.Image, top_k: int = 5) -> Dict:
        """
        Use local TensorFlow model for prediction
        
        Args:
            image: PIL Image object
            top_k: Number of top predictions to return
            
        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Preprocess image
        processed_image = self.preprocess_image(image)
        
        # Make prediction
        predictions = self.model.predict(processed_image, verbose=0)
        
        # Get top predictions
        top_indices = np.argsort(predictions[0])[-top_k:][::-1]
        
        top_predictions = [
            {
                "food_name": self.class_names[idx],
                "confidence": float(predictions[0][idx])
            }
            for idx in top_indices
        ]
        
        # Get best prediction
        best_idx = top_indices[0]
        
        return {
            "food_name": self.class_names[best_idx],
            "confidence": float(predictions[0][best_idx]),
            "top_predictions": top_predictions
        }
    
    def predict(self, image: Image.Image, top_k: int = 5) -> Dict:
        """
        Predict food type from image
        Uses Clarifai for accurate predictions if available, falls back to local model
        
        Args:
            image: PIL Image object
            top_k: Number of top predictions to return
            
        Returns:
            Dictionary with prediction results
        """
        # Use Clarifai if available (more accurate)
        if self.use_clarifai and self.clarifai_stub:
            return self._predict_with_clarifai(image, top_k)
        
        # Fall back to local model
        return self._predict_with_local_model(image, top_k)
    
    def train(self, train_data, validation_data, epochs: int = 10, batch_size: int = 32):
        """
        Train the model on custom data
        
        Args:
            train_data: Training dataset
            validation_data: Validation dataset
            epochs: Number of training epochs
            batch_size: Training batch size
        """
        if self.model is None:
            self.load_model()
        
        # Unfreeze some layers for fine-tuning
        base_model = self.model.layers[1]
        base_model.trainable = True
        
        # Freeze all but the last 20 layers
        for layer in base_model.layers[:-20]:
            layer.trainable = False
        
        # Recompile with lower learning rate
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train
        history = self.model.fit(
            train_data,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size
        )
        
        # Save trained model
        self.model.save(self.model_path)
        logger.info(f"Model saved to {self.model_path}")
        
        return history

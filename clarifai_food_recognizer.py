"""
Realistic Food Recognition using Clarifai's Pre-trained Food Model
This replaces the untrained local model with an actual working food recognition service
"""

import os
from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api.status import status_code_pb2
from PIL import Image
import io

class ClarifaiFoodRecognizer:
    """
    Use Clarifai's pre-trained food model for accurate predictions
    Free tier: 1000 operations/month
    Sign up at: https://clarifai.com
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize Clarifai client
        
        Args:
            api_key: Your Clarifai API key (get from https://clarifai.com)
        """
        self.api_key = api_key or os.getenv('CLARIFAI_API_KEY')
        
        if not self.api_key:
            raise ValueError(
                "Clarifai API key required. Get one at https://clarifai.com\n"
                "Then set: export CLARIFAI_API_KEY='your-key-here'"
            )
        
        # Setup channel
        channel = ClarifaiChannel.get_grpc_channel()
        self.stub = service_pb2_grpc.V2Stub(channel)
        self.metadata = (('authorization', f'Key {self.api_key}'),)
        
        # Use Clarifai's food model
        self.model_id = "food-item-recognition"
        self.model_version_id = "1d5fd481e0cf4826aa72ec3ff049e044"
    
    def predict(self, image: Image.Image):
        """
        Predict food from image using Clarifai
        
        Args:
            image: PIL Image
            
        Returns:
            Dictionary with prediction results
        """
        # Convert image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()
        
        # Make prediction request
        request = service_pb2.PostModelOutputsRequest(
            model_id=self.model_id,
            version_id=self.model_version_id,
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
        
        response = self.stub.PostModelOutputs(request, metadata=self.metadata)
        
        if response.status.code != status_code_pb2.SUCCESS:
            raise Exception(f"Clarifai API error: {response.status.description}")
        
        # Parse results
        concepts = response.outputs[0].data.concepts
        
        # Get top predictions
        top_predictions = [
            {
                'food_name': concept.name.title(),
                'confidence': concept.value
            }
            for concept in concepts[:10]
        ]
        
        # Main prediction
        main_prediction = top_predictions[0] if top_predictions else {
            'food_name': 'Unknown',
            'confidence': 0.0
        }
        
        return {
            'food_name': main_prediction['food_name'],
            'confidence': main_prediction['confidence'],
            'top_predictions': top_predictions
        }


# Installation instructions
SETUP_INSTRUCTIONS = """
To use Clarifai for accurate food recognition:

1. Install Clarifai SDK:
   pip install clarifai-grpc

2. Sign up for free API key:
   https://clarifai.com/signup

3. Get your API key from:
   https://clarifai.com/settings/security

4. Set environment variable:
   export CLARIFAI_API_KEY='your-key-here'

5. Update food_recognition.py to use ClarifaiFoodRecognizer

Free tier includes 1000 operations/month - perfect for testing!
"""

if __name__ == "__main__":
    print(SETUP_INSTRUCTIONS)

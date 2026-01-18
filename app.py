from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from typing import Optional
import numpy as np
from PIL import Image
import io
import tensorflow as tf
from tensorflow import keras
import logging

from database import DatabaseManager
from food_recognition import FoodRecognitionModel
from calorie_estimator import CalorieEstimator
from mobile_api import mobile_router, init_mobile_router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Food Recognition and Calorie Estimation API",
    description="AI-powered food recognition and calorie estimation service",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
db_manager = DatabaseManager()
food_model = FoodRecognitionModel()
calorie_estimator = CalorieEstimator()

# Include mobile API routes
app.include_router(mobile_router)


@app.on_event("startup")
async def startup_event():
    """Initialize database and models on startup"""
    logger.info("Starting Food Recognition API...")
    db_manager.create_tables()
    food_model.load_model()
    
    # Initialize mobile API router
    init_mobile_router(food_model, calorie_estimator, db_manager)
    
    logger.info("API ready!")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Food Recognition and Calorie Estimation API",
        "status": "active",
        "version": "1.0.0"
    }


@app.post("/predict")
async def predict_food(file: UploadFile = File(...)):
    """
    Predict food type and estimate calories from uploaded image
    
    Args:
        file: Uploaded image file
        
    Returns:
        JSON response with food prediction and calorie estimation
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Get food prediction
        prediction_result = food_model.predict(image)
        
        # Estimate calories
        calorie_info = calorie_estimator.estimate(
            food_name=prediction_result['food_name'],
            confidence=prediction_result['confidence']
        )
        
        # Combine results
        result = {
            "success": True,
            "food_name": prediction_result['food_name'],
            "confidence": round(prediction_result['confidence'] * 100, 2),
            "calories": calorie_info['calories'],
            "serving_size": calorie_info['serving_size'],
            "nutritional_info": calorie_info['nutritional_info'],
            "top_predictions": prediction_result.get('top_predictions', [])
        }
        
        # Save to database
        db_manager.save_prediction(
            food_name=result['food_name'],
            confidence=result['confidence'],
            calories=result['calories'],
            image_name=file.filename
        )
        
        logger.info(f"Prediction: {result['food_name']} ({result['confidence']}%)")
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.get("/history")
async def get_history(limit: int = 10):
    """
    Get prediction history
    
    Args:
        limit: Number of records to return
        
    Returns:
        List of past predictions
    """
    try:
        history = db_manager.get_history(limit=limit)
        return {"success": True, "history": history}
    except Exception as e:
        logger.error(f"Error fetching history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_statistics():
    """
    Get usage statistics
    
    Returns:
        Statistics about predictions
    """
    try:
        stats = db_manager.get_statistics()
        return {"success": True, "statistics": stats}
    except Exception as e:
        logger.error(f"Error fetching statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/history/{prediction_id}")
async def delete_prediction(prediction_id: int):
    """
    Delete a prediction from history
    
    Args:
        prediction_id: ID of prediction to delete
    """
    try:
        db_manager.delete_prediction(prediction_id)
        return {"success": True, "message": "Prediction deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

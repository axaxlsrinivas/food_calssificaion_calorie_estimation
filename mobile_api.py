"""
Mobile API Endpoints for Food Recognition
Optimized for mobile app integration (.NET MAUI, React Native, Flutter, etc.)
"""
from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import Optional, List
from pydantic import BaseModel
from PIL import Image
import io
import logging

logger = logging.getLogger(__name__)

# Create router for mobile endpoints
mobile_router = APIRouter(prefix="/api/mobile", tags=["Mobile"])


# Request/Response Models
class NutritionalInfo(BaseModel):
    protein: str
    carbs: str
    fat: str
    fiber: str
    sugar: str


class MobilePredictionResponse(BaseModel):
    success: bool
    food_name: str
    confidence: float
    calories: int
    serving_size: str
    nutritional_info: NutritionalInfo
    alternatives: List[dict]
    prediction_id: int


class HistoryItem(BaseModel):
    id: int
    food_name: str
    confidence: float
    calories: int
    timestamp: str


class MobileHistoryResponse(BaseModel):
    success: bool
    total_count: int
    items: List[HistoryItem]


class MobileStatsResponse(BaseModel):
    success: bool
    total_predictions: int
    today_predictions: int
    today_calories: int
    average_confidence: float
    most_common_food: Optional[str]


class DeleteResponse(BaseModel):
    success: bool
    message: str


# Import app components (will be injected)
food_model = None
calorie_estimator = None
db_manager = None


def init_mobile_router(model, estimator, db):
    """Initialize mobile router with app components"""
    global food_model, calorie_estimator, db_manager
    food_model = model
    calorie_estimator = estimator
    db_manager = db


@mobile_router.get("/health")
async def mobile_health_check():
    """
    Mobile-specific health check endpoint
    Returns API status and version info
    """
    return {
        "success": True,
        "status": "online",
        "version": "1.0.0",
        "endpoints": [
            "/api/mobile/predict",
            "/api/mobile/history",
            "/api/mobile/stats",
            "/api/mobile/delete/{id}"
        ]
    }


@mobile_router.post("/predict", response_model=MobilePredictionResponse)
async def mobile_predict_food(
    file: UploadFile = File(...),
    save_to_history: bool = Query(True, description="Save prediction to history")
):
    """
    Predict food from image - Mobile optimized
    
    Args:
        file: Image file (JPEG, PNG)
        save_to_history: Whether to save prediction to history (default: True)
        
    Returns:
        Mobile-friendly prediction response with flattened nutrition data
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
        
        # Prepare alternatives (top 5 predictions)
        alternatives = [
            {
                "food_name": pred['food_name'],
                "confidence": round(pred['confidence'] * 100, 2)
            }
            for pred in prediction_result.get('top_predictions', [])[1:6]  # Skip first (main result)
        ]
        
        # Save to database if requested
        prediction_id = 0
        if save_to_history:
            prediction_id = db_manager.save_prediction(
                food_name=prediction_result['food_name'],
                confidence=round(prediction_result['confidence'] * 100, 2),
                calories=calorie_info['calories'],
                image_name=file.filename
            )
        
        # Prepare nutritional info
        nutrition = calorie_info['nutritional_info']
        
        response = MobilePredictionResponse(
            success=True,
            food_name=prediction_result['food_name'],
            confidence=round(prediction_result['confidence'] * 100, 2),
            calories=calorie_info['calories'],
            serving_size=calorie_info['serving_size'],
            nutritional_info=NutritionalInfo(
                protein=nutrition['protein'],
                carbs=nutrition['carbs'],
                fat=nutrition['fat'],
                fiber=nutrition['fiber'],
                sugar=nutrition['sugar']
            ),
            alternatives=alternatives,
            prediction_id=prediction_id
        )
        
        logger.info(f"Mobile prediction: {response.food_name} ({response.confidence}%)")
        
        return response
        
    except Exception as e:
        logger.error(f"Mobile prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@mobile_router.get("/history", response_model=MobileHistoryResponse)
async def mobile_get_history(
    limit: int = Query(20, ge=1, le=100, description="Number of items to return"),
    offset: int = Query(0, ge=0, description="Number of items to skip")
):
    """
    Get prediction history - Mobile optimized with pagination
    
    Args:
        limit: Maximum number of items to return (1-100, default: 20)
        offset: Number of items to skip for pagination (default: 0)
        
    Returns:
        Paginated history with total count
    """
    try:
        # Get history from database
        history = db_manager.get_history(limit=limit)
        
        # Get total count (for pagination)
        # Note: You may want to add a count method to DatabaseManager
        
        # Format for mobile
        items = [
            HistoryItem(
                id=item['id'],
                food_name=item['food_name'],
                confidence=item['confidence'],
                calories=item['calories'],
                timestamp=item['created_at']
            )
            for item in history
        ]
        
        response = MobileHistoryResponse(
            success=True,
            total_count=len(items),
            items=items
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Mobile history error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@mobile_router.get("/stats", response_model=MobileStatsResponse)
async def mobile_get_stats():
    """
    Get statistics - Mobile optimized
    
    Returns:
        Simplified statistics for mobile display
    """
    try:
        stats = db_manager.get_statistics()
        
        # Get most common food
        most_common = None
        if stats['top_foods'] and len(stats['top_foods']) > 0:
            most_common = stats['top_foods'][0]['food_name']
        
        response = MobileStatsResponse(
            success=True,
            total_predictions=stats['total_predictions'],
            today_predictions=stats['today_predictions'],
            today_calories=stats['today_calories'],
            average_confidence=stats['average_confidence'],
            most_common_food=most_common
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Mobile stats error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@mobile_router.delete("/delete/{prediction_id}", response_model=DeleteResponse)
async def mobile_delete_prediction(prediction_id: int):
    """
    Delete a prediction from history
    
    Args:
        prediction_id: ID of the prediction to delete
        
    Returns:
        Success confirmation
    """
    try:
        db_manager.delete_prediction(prediction_id)
        
        return DeleteResponse(
            success=True,
            message=f"Prediction {prediction_id} deleted successfully"
        )
        
    except Exception as e:
        logger.error(f"Mobile delete error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@mobile_router.get("/foods")
async def mobile_get_available_foods():
    """
    Get list of all foods the API can recognize
    
    Returns:
        List of food names and their complete nutritional information
    """
    try:
        foods = calorie_estimator.get_all_foods()
        
        # Get nutrition data for each food
        food_list = []
        for food_name in foods:
            calorie_info = calorie_estimator.estimate(food_name, confidence=1.0)
            nutrition = calorie_info['nutritional_info']
            food_list.append({
                "name": food_name,
                "calories": calorie_info['calories'],
                "serving_size": calorie_info['serving_size'],
                "nutritional_info": {
                    "protein": nutrition['protein'],
                    "carbs": nutrition['carbs'],
                    "fat": nutrition['fat'],
                    "fiber": nutrition['fiber'],
                    "sugar": nutrition['sugar']
                }
            })
        
        return {
            "success": True,
            "total_foods": len(food_list),
            "foods": sorted(food_list, key=lambda x: x['name'])
        }
        
    except Exception as e:
        logger.error(f"Mobile foods list error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@mobile_router.get("/search")
async def mobile_search_food(
    query: str = Query(..., min_length=1, description="Search query")
):
    """
    Search for food by name
    
    Args:
        query: Search term
        
    Returns:
        Matching foods with nutrition info
    """
    try:
        all_foods = calorie_estimator.get_all_foods()
        
        # Simple case-insensitive search
        query_lower = query.lower()
        matching_foods = [
            food for food in all_foods 
            if query_lower in food.lower()
        ]
        
        # Get details for matching foods
        results = []
        for food_name in matching_foods:
            calorie_info = calorie_estimator.estimate(food_name, confidence=1.0)
            nutrition = calorie_info['nutritional_info']
            results.append({
                "name": food_name,
                "calories": calorie_info['calories'],
                "serving_size": calorie_info['serving_size'],
                "nutritional_info": {
                    "protein": nutrition['protein'],
                    "carbs": nutrition['carbs'],
                    "fat": nutrition['fat'],
                    "fiber": nutrition['fiber'],
                    "sugar": nutrition['sugar']
                }
            })
        
        return {
            "success": True,
            "query": query,
            "results_count": len(results),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Mobile search error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

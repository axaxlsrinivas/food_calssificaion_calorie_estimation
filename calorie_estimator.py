import json
import os
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class CalorieEstimator:
    """
    Estimate calories and nutritional information for recognized foods
    """
    
    def __init__(self, database_path: str = "data/nutrition_database.json"):
        self.database_path = database_path
        self.nutrition_db = {}
        self._load_nutrition_database()
    
    def _load_nutrition_database(self):
        """Load nutrition database from file or create default"""
        if os.path.exists(self.database_path):
            with open(self.database_path, 'r') as f:
                self.nutrition_db = json.load(f)
            logger.info(f"Loaded nutrition database with {len(self.nutrition_db)} entries")
        else:
            # Create default nutrition database
            self._create_default_database()
            logger.info("Created default nutrition database")
    
    def _create_default_database(self):
        """
        Create a default nutrition database
        Values are per standard serving size
        """
        self.nutrition_db = {
            "Apple": {
                "calories": 95,
                "serving_size": "1 medium (182g)",
                "protein": 0.5,
                "carbs": 25,
                "fat": 0.3,
                "fiber": 4.4,
                "sugar": 19
            },
            "Banana": {
                "calories": 105,
                "serving_size": "1 medium (118g)",
                "protein": 1.3,
                "carbs": 27,
                "fat": 0.4,
                "fiber": 3.1,
                "sugar": 14
            },
            "Bread": {
                "calories": 79,
                "serving_size": "1 slice (28g)",
                "protein": 2.7,
                "carbs": 15,
                "fat": 1,
                "fiber": 0.8,
                "sugar": 1.5
            },
            "Burger": {
                "calories": 354,
                "serving_size": "1 burger (143g)",
                "protein": 21,
                "carbs": 30,
                "fat": 16,
                "fiber": 1.5,
                "sugar": 5
            },
            "Cake": {
                "calories": 257,
                "serving_size": "1 slice (80g)",
                "protein": 3,
                "carbs": 38,
                "fat": 10,
                "fiber": 0.6,
                "sugar": 24
            },
            "Chicken": {
                "calories": 165,
                "serving_size": "100g (cooked)",
                "protein": 31,
                "carbs": 0,
                "fat": 3.6,
                "fiber": 0,
                "sugar": 0
            },
            "Coffee": {
                "calories": 2,
                "serving_size": "1 cup (240ml)",
                "protein": 0.3,
                "carbs": 0,
                "fat": 0,
                "fiber": 0,
                "sugar": 0
            },
            "Cookie": {
                "calories": 49,
                "serving_size": "1 cookie (10g)",
                "protein": 0.5,
                "carbs": 6.6,
                "fat": 2.3,
                "fiber": 0.2,
                "sugar": 3.5
            },
            "Donut": {
                "calories": 269,
                "serving_size": "1 donut (66g)",
                "protein": 3.1,
                "carbs": 31,
                "fat": 15,
                "fiber": 0.9,
                "sugar": 12
            },
            "Egg": {
                "calories": 78,
                "serving_size": "1 large egg (50g)",
                "protein": 6.3,
                "carbs": 0.6,
                "fat": 5.3,
                "fiber": 0,
                "sugar": 0.6
            },
            "French Fries": {
                "calories": 312,
                "serving_size": "100g",
                "protein": 3.4,
                "carbs": 41,
                "fat": 15,
                "fiber": 3.8,
                "sugar": 0.3
            },
            "Grape": {
                "calories": 69,
                "serving_size": "1 cup (151g)",
                "protein": 0.7,
                "carbs": 18,
                "fat": 0.2,
                "fiber": 0.9,
                "sugar": 15
            },
            "Hot Dog": {
                "calories": 151,
                "serving_size": "1 hot dog (45g)",
                "protein": 5.1,
                "carbs": 2,
                "fat": 13,
                "fiber": 0,
                "sugar": 1
            },
            "Ice Cream": {
                "calories": 207,
                "serving_size": "1 cup (132g)",
                "protein": 3.5,
                "carbs": 24,
                "fat": 11,
                "fiber": 0.7,
                "sugar": 21
            },
            "Orange": {
                "calories": 62,
                "serving_size": "1 medium (131g)",
                "protein": 1.2,
                "carbs": 15,
                "fat": 0.2,
                "fiber": 3.1,
                "sugar": 12
            },
            "Pancake": {
                "calories": 86,
                "serving_size": "1 pancake (38g)",
                "protein": 2.4,
                "carbs": 11,
                "fat": 3.5,
                "fiber": 0.5,
                "sugar": 2
            },
            "Pizza": {
                "calories": 285,
                "serving_size": "1 slice (107g)",
                "protein": 12,
                "carbs": 36,
                "fat": 10,
                "fiber": 2.5,
                "sugar": 4
            },
            "Rice": {
                "calories": 130,
                "serving_size": "1 cup cooked (158g)",
                "protein": 2.7,
                "carbs": 28,
                "fat": 0.3,
                "fiber": 0.4,
                "sugar": 0.1
            },
            "Salad": {
                "calories": 33,
                "serving_size": "1 cup (47g)",
                "protein": 1.2,
                "carbs": 6.3,
                "fat": 0.2,
                "fiber": 2.1,
                "sugar": 2.4
            },
            "Sandwich": {
                "calories": 304,
                "serving_size": "1 sandwich (146g)",
                "protein": 13,
                "carbs": 39,
                "fat": 10,
                "fiber": 4,
                "sugar": 6
            },
            "Spaghetti": {
                "calories": 221,
                "serving_size": "1 cup cooked (140g)",
                "protein": 8.1,
                "carbs": 43,
                "fat": 1.3,
                "fiber": 2.5,
                "sugar": 0.8
            },
            "Steak": {
                "calories": 271,
                "serving_size": "100g (cooked)",
                "protein": 25,
                "carbs": 0,
                "fat": 19,
                "fiber": 0,
                "sugar": 0
            },
            "Strawberry": {
                "calories": 49,
                "serving_size": "1 cup (152g)",
                "protein": 1,
                "carbs": 12,
                "fat": 0.5,
                "fiber": 3,
                "sugar": 7.4
            },
            "Sushi": {
                "calories": 145,
                "serving_size": "1 roll (6 pieces)",
                "protein": 6,
                "carbs": 24,
                "fat": 3.7,
                "fiber": 3.5,
                "sugar": 3
            },
            "Taco": {
                "calories": 226,
                "serving_size": "1 taco (102g)",
                "protein": 10,
                "carbs": 20,
                "fat": 12,
                "fiber": 3.2,
                "sugar": 1.8            },
            "Meat": {
                "calories": 250,
                "serving_size": "100g (cooked)",
                "protein": 26,
                "carbs": 0,
                "fat": 15,
                "fiber": 0,
                "sugar": 0
            },
            "Beef": {
                "calories": 250,
                "serving_size": "100g (cooked)",
                "protein": 26,
                "carbs": 0,
                "fat": 15,
                "fiber": 0,
                "sugar": 0
            },
            "Pork": {
                "calories": 242,
                "serving_size": "100g (cooked)",
                "protein": 27,
                "carbs": 0,
                "fat": 14,
                "fiber": 0,
                "sugar": 0
            },
            "Lamb": {
                "calories": 294,
                "serving_size": "100g (cooked)",
                "protein": 25,
                "carbs": 0,
                "fat": 21,
                "fiber": 0,
                "sugar": 0
            },
            "Mutton": {
                "calories": 294,
                "serving_size": "100g (cooked)",
                "protein": 25,
                "carbs": 0,
                "fat": 21,
                "fiber": 0,
                "sugar": 0
            },
            "Fish": {
                "calories": 206,
                "serving_size": "100g (cooked)",
                "protein": 22,
                "carbs": 0,
                "fat": 12,
                "fiber": 0,
                "sugar": 0
            },
            "Salmon": {
                "calories": 206,
                "serving_size": "100g (cooked)",
                "protein": 22,
                "carbs": 0,
                "fat": 12,
                "fiber": 0,
                "sugar": 0
            },
            "Tuna": {
                "calories": 184,
                "serving_size": "100g (cooked)",
                "protein": 30,
                "carbs": 0,
                "fat": 6.3,
                "fiber": 0,
                "sugar": 0
            },
            "Shrimp": {
                "calories": 99,
                "serving_size": "100g (cooked)",
                "protein": 24,
                "carbs": 0.2,
                "fat": 0.3,
                "fiber": 0,
                "sugar": 0
            },
            "Cheese": {
                "calories": 402,
                "serving_size": "100g",
                "protein": 25,
                "carbs": 1.3,
                "fat": 33,
                "fiber": 0,
                "sugar": 0.5
            },
            "Mozzarella": {
                "calories": 280,
                "serving_size": "100g",
                "protein": 28,
                "carbs": 3.1,
                "fat": 17,
                "fiber": 0,
                "sugar": 1.2
            },
            "Pepperoni": {
                "calories": 494,
                "serving_size": "100g",
                "protein": 23,
                "carbs": 4,
                "fat": 44,
                "fiber": 0,
                "sugar": 1
            },
            "Salami": {
                "calories": 407,
                "serving_size": "100g",
                "protein": 22,
                "carbs": 1.6,
                "fat": 34,
                "fiber": 0,
                "sugar": 0
            },
            "Bacon": {
                "calories": 541,
                "serving_size": "100g (cooked)",
                "protein": 37,
                "carbs": 1.4,
                "fat": 42,
                "fiber": 0,
                "sugar": 0
            },
            "Pasta": {
                "calories": 131,
                "serving_size": "100g (cooked)",
                "protein": 5,
                "carbs": 25,
                "fat": 1.1,
                "fiber": 1.8,
                "sugar": 0.6
            },
            "Noodles": {
                "calories": 138,
                "serving_size": "100g (cooked)",
                "protein": 4.5,
                "carbs": 25,
                "fat": 2.1,
                "fiber": 1.2,
                "sugar": 0.4
            },
            "Soup": {
                "calories": 38,
                "serving_size": "1 cup (240ml)",
                "protein": 2,
                "carbs": 5,
                "fat": 1.2,
                "fiber": 0.5,
                "sugar": 1.5
            },
            "Vegetables": {
                "calories": 65,
                "serving_size": "1 cup (150g)",
                "protein": 2.5,
                "carbs": 13,
                "fat": 0.5,
                "fiber": 4,
                "sugar": 6
            },
            "Broccoli": {
                "calories": 55,
                "serving_size": "1 cup (156g)",
                "protein": 4,
                "carbs": 11,
                "fat": 0.6,
                "fiber": 5,
                "sugar": 2.6
            },
            "Carrot": {
                "calories": 52,
                "serving_size": "1 cup (128g)",
                "protein": 1.2,
                "carbs": 12,
                "fat": 0.3,
                "fiber": 3.6,
                "sugar": 6.1
            },
            "Potato": {
                "calories": 161,
                "serving_size": "1 medium (173g)",
                "protein": 4.3,
                "carbs": 37,
                "fat": 0.2,
                "fiber": 4.7,
                "sugar": 1.9
            },
            "Tomato": {
                "calories": 22,
                "serving_size": "1 medium (123g)",
                "protein": 1.1,
                "carbs": 4.8,
                "fat": 0.2,
                "fiber": 1.5,
                "sugar": 3.2
            },
            "Cucumber": {
                "calories": 16,
                "serving_size": "1 cup (104g)",
                "protein": 0.7,
                "carbs": 3.6,
                "fat": 0.1,
                "fiber": 0.5,
                "sugar": 1.7
            },
            "Onion": {
                "calories": 44,
                "serving_size": "1 medium (110g)",
                "protein": 1.2,
                "carbs": 10,
                "fat": 0.1,
                "fiber": 1.9,
                "sugar": 4.7            }
        }
        
        # Save to file
        os.makedirs(os.path.dirname(self.database_path), exist_ok=True)
        with open(self.database_path, 'w') as f:
            json.dump(self.nutrition_db, f, indent=2)
    
    def estimate(self, food_name: str, confidence: float = 1.0) -> Dict:
        """
        Estimate calories and nutritional information for a food item
        
        Args:
            food_name: Name of the food
            confidence: Confidence level of the prediction
            
        Returns:
            Dictionary with nutritional information
        """
        # Get nutrition data (case-insensitive)
        nutrition_data = None
        for key in self.nutrition_db.keys():
            if key.lower() == food_name.lower():
                nutrition_data = self.nutrition_db[key]
                break
        
        if nutrition_data is None:
            # Return estimated values if food not in database
            logger.warning(f"Food '{food_name}' not in database. Using estimates.")
            return {
                "calories": 200,  # Default estimate
                "serving_size": "1 serving",
                "nutritional_info": {
                    "protein": "N/A",
                    "carbs": "N/A",
                    "fat": "N/A",
                    "fiber": "N/A",
                    "sugar": "N/A"
                },
                "note": "Estimated values - food not in database"
            }
        
        return {
            "calories": nutrition_data["calories"],
            "serving_size": nutrition_data["serving_size"],
            "nutritional_info": {
                "protein": f"{nutrition_data['protein']}g",
                "carbs": f"{nutrition_data['carbs']}g",
                "fat": f"{nutrition_data['fat']}g",
                "fiber": f"{nutrition_data['fiber']}g",
                "sugar": f"{nutrition_data['sugar']}g"
            }
        }
    
    def add_food(self, food_name: str, nutrition_data: Dict):
        """
        Add or update a food item in the database
        
        Args:
            food_name: Name of the food
            nutrition_data: Nutritional information dictionary
        """
        self.nutrition_db[food_name] = nutrition_data
        
        # Save to file
        with open(self.database_path, 'w') as f:
            json.dump(self.nutrition_db, f, indent=2)
        
        logger.info(f"Added/updated food: {food_name}")
    
    def get_all_foods(self) -> list:
        """Get list of all foods in database"""
        return list(self.nutrition_db.keys())

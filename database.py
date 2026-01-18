import sqlite3
from datetime import datetime
from typing import List, Dict, Optional
import logging
import os

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Database manager for storing predictions and user data
    Uses SQLite for simplicity - can be upgraded to PostgreSQL/MySQL for production
    """
    
    def __init__(self, db_path: str = "data/food_recognition.db"):
        self.db_path = db_path
        self._ensure_data_directory()
    
    def _ensure_data_directory(self):
        """Create data directory if it doesn't exist"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
    
    def get_connection(self) -> sqlite3.Connection:
        """Get database connection"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def create_tables(self):
        """Create database tables if they don't exist"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Predictions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                food_name TEXT NOT NULL,
                confidence REAL NOT NULL,
                calories INTEGER NOT NULL,
                image_name TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # User statistics table (for future use)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE NOT NULL,
                total_predictions INTEGER DEFAULT 0,
                total_calories INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(date)
            )
        ''')
        
        # Food feedback table (for model improvement)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id INTEGER,
                is_correct BOOLEAN,
                correct_food_name TEXT,
                user_comment TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (prediction_id) REFERENCES predictions(id)
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Database tables created successfully")
    
    def save_prediction(
        self,
        food_name: str,
        confidence: float,
        calories: int,
        image_name: Optional[str] = None
    ) -> int:
        """
        Save a prediction to the database
        
        Args:
            food_name: Predicted food name
            confidence: Prediction confidence
            calories: Estimated calories
            image_name: Original image filename
            
        Returns:
            ID of the saved prediction
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO predictions (food_name, confidence, calories, image_name)
            VALUES (?, ?, ?, ?)
        ''', (food_name, confidence, calories, image_name))
        
        prediction_id = cursor.lastrowid
        
        # Update daily statistics
        today = datetime.now().date()
        cursor.execute('''
            INSERT INTO user_stats (date, total_predictions, total_calories)
            VALUES (?, 1, ?)
            ON CONFLICT(date) DO UPDATE SET
                total_predictions = total_predictions + 1,
                total_calories = total_calories + ?
        ''', (today, calories, calories))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Saved prediction: {food_name} (ID: {prediction_id})")
        return prediction_id
    
    def get_history(self, limit: int = 10) -> List[Dict]:
        """
        Get prediction history
        
        Args:
            limit: Maximum number of records to return
            
        Returns:
            List of prediction dictionaries
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, food_name, confidence, calories, image_name, created_at
            FROM predictions
            ORDER BY created_at DESC
            LIMIT ?
        ''', (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def get_statistics(self) -> Dict:
        """
        Get overall statistics
        
        Returns:
            Dictionary with statistics
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Total predictions
        cursor.execute('SELECT COUNT(*) as total FROM predictions')
        total_predictions = cursor.fetchone()['total']
        
        # Today's statistics
        today = datetime.now().date()
        cursor.execute('''
            SELECT total_predictions, total_calories
            FROM user_stats
            WHERE date = ?
        ''', (today,))
        
        today_stats = cursor.fetchone()
        today_predictions = today_stats['total_predictions'] if today_stats else 0
        today_calories = today_stats['total_calories'] if today_stats else 0
        
        # Most common foods
        cursor.execute('''
            SELECT food_name, COUNT(*) as count
            FROM predictions
            GROUP BY food_name
            ORDER BY count DESC
            LIMIT 5
        ''')
        
        top_foods = [dict(row) for row in cursor.fetchall()]
        
        # Average confidence
        cursor.execute('SELECT AVG(confidence) as avg_confidence FROM predictions')
        avg_confidence = cursor.fetchone()['avg_confidence'] or 0
        
        conn.close()
        
        return {
            "total_predictions": total_predictions,
            "today_predictions": today_predictions,
            "today_calories": today_calories,
            "average_confidence": round(avg_confidence, 2),
            "top_foods": top_foods
        }
    
    def delete_prediction(self, prediction_id: int):
        """
        Delete a prediction from the database
        
        Args:
            prediction_id: ID of prediction to delete
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM predictions WHERE id = ?', (prediction_id,))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Deleted prediction ID: {prediction_id}")
    
    def save_feedback(
        self,
        prediction_id: int,
        is_correct: bool,
        correct_food_name: Optional[str] = None,
        user_comment: Optional[str] = None
    ):
        """
        Save user feedback for a prediction
        
        Args:
            prediction_id: ID of the prediction
            is_correct: Whether the prediction was correct
            correct_food_name: Correct food name if prediction was wrong
            user_comment: Additional user comments
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO feedback (prediction_id, is_correct, correct_food_name, user_comment)
            VALUES (?, ?, ?, ?)
        ''', (prediction_id, is_correct, correct_food_name, user_comment))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Saved feedback for prediction ID: {prediction_id}")


# Database Schema Documentation
"""
RECOMMENDED DATABASE OPTIONS:

1. SQLite (Current Implementation) - Good for:
   - Development and testing
   - Small to medium-scale deployments
   - Single-server applications
   - Simple setup with no external dependencies

2. PostgreSQL (Recommended for Production) - Good for:
   - Large-scale applications
   - Multiple concurrent users
   - Complex queries and analytics
   - Better data integrity and ACID compliance
   - JSON/JSONB support for flexible data
   Migration: Replace sqlite3 with psycopg2

3. MySQL/MariaDB - Good for:
   - High-performance read operations
   - Web applications
   - Good community support
   Migration: Replace sqlite3 with mysql-connector-python

4. MongoDB (NoSQL Option) - Good for:
   - Flexible schema requirements
   - Rapid development
   - Document-based storage
   - Horizontal scaling
   Migration: Replace with pymongo

For production, I recommend PostgreSQL for its reliability,
performance, and advanced features.
"""

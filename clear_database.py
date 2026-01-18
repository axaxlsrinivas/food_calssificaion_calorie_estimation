"""
Clear all data from the database
"""
import os
import sqlite3
from database import DatabaseManager

def clear_database():
    """Clear all data from the database tables"""
    
    db_path = "data/food_recognition.db"
    
    if not os.path.exists(db_path):
        print("✓ Database doesn't exist. Nothing to clear.")
        return
    
    try:
        print("Clearing database...")
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Delete all records from tables
        cursor.execute("DELETE FROM predictions")
        deleted_predictions = cursor.rowcount
        
        cursor.execute("DELETE FROM user_stats")
        deleted_stats = cursor.rowcount
        
        cursor.execute("DELETE FROM feedback")
        deleted_feedback = cursor.rowcount
        
        # Reset auto-increment counters
        cursor.execute("DELETE FROM sqlite_sequence WHERE name='predictions'")
        cursor.execute("DELETE FROM sqlite_sequence WHERE name='user_stats'")
        cursor.execute("DELETE FROM sqlite_sequence WHERE name='feedback'")
        
        conn.commit()
        conn.close()
        
        print("\n" + "="*60)
        print("✓ Database Cleared Successfully!")
        print("="*60)
        print(f"  Deleted {deleted_predictions} predictions")
        print(f"  Deleted {deleted_stats} user stats")
        print(f"  Deleted {deleted_feedback} feedback entries")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Error clearing database: {e}")

def delete_database():
    """Completely delete the database file"""
    
    db_path = "data/food_recognition.db"
    
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
            print("✓ Database file deleted completely!")
            print("  A new database will be created when the API starts.")
        except Exception as e:
            print(f"❌ Error deleting database file: {e}")
    else:
        print("✓ Database file doesn't exist.")

if __name__ == "__main__":
    import sys
    
    print("Database Cleanup Utility")
    print("="*60)
    
    if len(sys.argv) > 1 and sys.argv[1] == "--delete":
        print("\nCompletely deleting database file...\n")
        delete_database()
    else:
        print("\nClearing all data from database...\n")
        clear_database()
        print("\nTo completely delete the database file, run:")
        print("  python clear_database.py --delete")

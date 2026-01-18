"""
Simple test client for the Food Recognition API
"""
import requests
import json
from pathlib import Path


def test_health_check():
    """Test the health check endpoint"""
    print("Testing health check endpoint...")
    response = requests.get("http://localhost:8000/")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()


def test_predict(image_path: str):
    """Test the prediction endpoint"""
    print(f"Testing prediction with image: {image_path}")
    
    if not Path(image_path).exists():
        print(f"Error: Image file not found: {image_path}")
        return
    
    with open(image_path, 'rb') as f:
        files = {'file': f}
        response = requests.post("http://localhost:8000/predict", files=files)
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"\nResults:")
        print(f"  Food: {result['food_name']}")
        print(f"  Confidence: {result['confidence']}%")
        print(f"  Calories: {result['calories']}")
        print(f"  Serving Size: {result['serving_size']}")
        print(f"\n  Nutritional Info:")
        for key, value in result['nutritional_info'].items():
            print(f"    {key.capitalize()}: {value}")
        print(f"\n  Top Predictions:")
        for pred in result['top_predictions'][:3]:
            print(f"    - {pred['food_name']}: {pred['confidence']*100:.2f}%")
    else:
        print(f"Error: {response.text}")
    print()


def test_history():
    """Test the history endpoint"""
    print("Testing history endpoint...")
    response = requests.get("http://localhost:8000/history?limit=5")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"\nRecent Predictions ({len(result['history'])} items):")
        for item in result['history']:
            print(f"  - {item['food_name']} ({item['confidence']}%) - {item['calories']} cal")
    print()


def test_statistics():
    """Test the statistics endpoint"""
    print("Testing statistics endpoint...")
    response = requests.get("http://localhost:8000/stats")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        stats = result['statistics']
        print(f"\nStatistics:")
        print(f"  Total Predictions: {stats['total_predictions']}")
        print(f"  Today's Predictions: {stats['today_predictions']}")
        print(f"  Today's Calories: {stats['today_calories']}")
        print(f"  Average Confidence: {stats['average_confidence']}%")
        print(f"\n  Top Foods:")
        for food in stats['top_foods']:
            print(f"    - {food['food_name']}: {food['count']} times")
    print()


if __name__ == "__main__":
    print("=" * 60)
    print("Food Recognition API Test Client")
    print("=" * 60)
    print("\nMake sure the API server is running on http://localhost:8000")
    print("Start the server with: python app.py\n")
    
    # Test health check
    try:
        test_health_check()
    except requests.exceptions.ConnectionError:
        print("ERROR: Cannot connect to API. Make sure the server is running!")
        exit(1)
    
    # Test with sample image (if provided)
    import sys
    if len(sys.argv) > 1:
        test_predict(sys.argv[1])
    else:
        print("To test prediction, provide an image path:")
        print("  python test_client.py path/to/image.jpg\n")
    
    # Test other endpoints
    test_history()
    test_statistics()
    
    print("=" * 60)
    print("Testing complete!")
    print("=" * 60)

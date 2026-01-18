"""
QUICK START GUIDE - Food Recognition API
========================================

Follow these steps to get your Food Recognition and Calorie Estimation API running!

## Step 1: Verify Python Environment ✓

Your Python environment is already configured!
- Python Version: 3.9.6
- Virtual Environment: Activated
- Location: /Users/cr63494/Documents/GIETPOC/.venv

## Step 2: Install Dependencies

Run this command to install all required packages:

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install fastapi uvicorn[standard] python-multipart tensorflow pillow numpy pandas
```

## Step 3: Start the Server

Run the API server:

```bash
python app.py
```

Or using uvicorn directly:

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

The server will start at: http://localhost:8000

## Step 4: Test the API

### Option 1: Interactive API Documentation
Open your browser and go to:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Option 2: Using the Test Client
```bash
python test_client.py
```

With an image:
```bash
python test_client.py path/to/your/food_image.jpg
```

### Option 3: Using cURL
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_food_image.jpg"
```

## Step 5: Available Endpoints

1. **Health Check**
   - GET http://localhost:8000/
   - Check if API is running

2. **Predict Food**
   - POST http://localhost:8000/predict
   - Upload image to recognize food and get calories

3. **View History**
   - GET http://localhost:8000/history?limit=10
   - See past predictions

4. **View Statistics**
   - GET http://localhost:8000/stats
   - Get usage statistics

5. **Delete Prediction**
   - DELETE http://localhost:8000/history/{id}
   - Remove a prediction

## Understanding the Project Structure

```
GIETPOC/
├── app.py                      # Main FastAPI application (START HERE)
├── food_recognition.py         # AI model for food recognition
├── calorie_estimator.py        # Calorie and nutrition calculator
├── database.py                 # Database operations (SQLite)
├── test_client.py              # Test the API
├── requirements.txt            # Python dependencies
├── README.md                   # Full documentation
├── DATABASE_RECOMMENDATIONS.md # Database options guide
│
├── data/                       # Created automatically
│   ├── nutrition_database.json # Nutrition info for 25 foods
│   └── food_recognition.db     # SQLite database
│
└── models/                     # Created automatically
    ├── food_model.h5           # Trained AI model (auto-downloaded)
    └── class_names.json        # List of food categories
```

## Features Implemented

✅ **Image Recognition**
   - Deep learning model (MobileNetV2)
   - Recognizes 25+ common foods
   - Returns confidence scores
   - Shows top 5 predictions

✅ **Calorie Estimation**
   - Detailed nutritional information
   - Protein, carbs, fat, fiber, sugar
   - Standard serving sizes
   - Expandable food database

✅ **Database Storage**
   - SQLite for easy setup
   - Prediction history tracking
   - Daily statistics
   - User feedback system

✅ **REST API**
   - Fast and efficient
   - Auto-generated documentation
   - CORS enabled
   - Error handling

## Common Use Cases

### 1. Recognize Food from Image
```python
import requests

url = "http://localhost:8000/predict"
files = {"file": open("pizza.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

### 2. Track Daily Calories
```python
# Get today's statistics
response = requests.get("http://localhost:8000/stats")
stats = response.json()
print(f"Today's calories: {stats['statistics']['today_calories']}")
```

### 3. View Recent Meals
```python
response = requests.get("http://localhost:8000/history?limit=10")
history = response.json()
for meal in history['history']:
    print(f"{meal['food_name']}: {meal['calories']} cal")
```

## Customization Options

### Add New Foods
Edit `calorie_estimator.py` to add new food items:
```python
calorie_estimator.add_food("Smoothie", {
    "calories": 150,
    "serving_size": "1 cup (240ml)",
    "protein": 5,
    "carbs": 30,
    "fat": 2,
    "fiber": 3,
    "sugar": 20
})
```

### Train Custom Model
To recognize additional foods, prepare a dataset and train:
```python
from food_recognition import FoodRecognitionModel

model = FoodRecognitionModel()
model.load_model()
model.train(train_dataset, validation_dataset, epochs=10)
```

### Change Database
See DATABASE_RECOMMENDATIONS.md for:
- PostgreSQL (recommended for production)
- MySQL/MariaDB
- MongoDB
- Cloud databases

## Troubleshooting

### Issue: Module not found
Solution: Install dependencies
```bash
pip install -r requirements.txt
```

### Issue: Port already in use
Solution: Change port in app.py
```python
uvicorn.run("app:app", port=8080)  # Use different port
```

### Issue: Model loading slow
Solution: This is normal for first run. TensorFlow downloads the model.

### Issue: Low confidence predictions
Solution: The default model needs training on your specific foods.
Consider fine-tuning with your own dataset.

## Production Deployment

### 1. Use Production Server
```bash
pip install gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker app:app
```

### 2. Environment Variables
Create `.env` file:
```
DB_HOST=your_db_host
DB_PASSWORD=your_password
API_KEY=your_api_key
```

### 3. Docker Deployment
```dockerfile
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 4. Cloud Deployment Options
- **Heroku**: Easy deployment, free tier available
- **AWS EC2**: Full control, scalable
- **Google Cloud Run**: Serverless, pay per use
- **Azure App Service**: Integrated with Azure services

## Performance Tips

1. **Use caching**: Implement Redis for frequent requests
2. **Optimize images**: Resize before uploading (max 800x800)
3. **Database indexing**: Already implemented
4. **Load balancing**: Use nginx for multiple instances
5. **CDN**: Store images in cloud storage (S3, GCS)

## Next Steps

1. ✅ API is ready to use
2. 📸 Try uploading food images
3. 📊 Check statistics and history
4. 🔧 Customize food database
5. 🚀 Deploy to production
6. 📱 Build mobile/web frontend

## Support & Documentation

- Full API docs: http://localhost:8000/docs
- README.md: Complete documentation
- DATABASE_RECOMMENDATIONS.md: Database guide
- Test with: python test_client.py

## Example Integration

### React Frontend
```javascript
const uploadImage = async (file) => {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await fetch('http://localhost:8000/predict', {
    method: 'POST',
    body: formData
  });
  
  const data = await response.json();
  console.log(`${data.food_name}: ${data.calories} calories`);
};
```

### Mobile App (React Native)
```javascript
const FormData = require('form-data');

const predictFood = async (imageUri) => {
  const formData = new FormData();
  formData.append('file', {
    uri: imageUri,
    type: 'image/jpeg',
    name: 'food.jpg'
  });
  
  const response = await fetch('http://your-api.com/predict', {
    method: 'POST',
    body: formData
  });
  
  return await response.json();
};
```

---

🎉 You're all set! Start the server with `python app.py` and begin recognizing food!

For questions or issues, check the README.md or the API documentation.
"""

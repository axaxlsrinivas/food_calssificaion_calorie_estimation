# Food Recognition and Calorie Estimation API

A web service built with Python for recognizing food from images and estimating calorie content using deep learning.

## Features

- 🍕 **Food Recognition**: Identify food items from images using deep learning
- 📊 **Calorie Estimation**: Get detailed nutritional information including calories, protein, carbs, and more
- 💾 **Database Storage**: Track prediction history and statistics
- 🚀 **Fast API**: High-performance REST API built with FastAPI
- 📈 **Analytics**: View usage statistics and trends

## Technology Stack

- **Framework**: FastAPI
- **Machine Learning**: TensorFlow/Keras with MobileNetV2
- **Database**: SQLite (upgradeable to PostgreSQL/MySQL)
- **Image Processing**: Pillow (PIL)
- **API Documentation**: Swagger UI (auto-generated)

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Install Dependencies**

```bash
pip install -r requirements.txt
```

2. **Run the Application**

```bash
python app.py
```

The API will be available at `http://localhost:8000`

## API Endpoints

### 1. Health Check
```
GET /
```
Check if the API is running.

### 2. Predict Food
```
POST /predict
```
Upload an image to recognize food and get calorie information.

**Request**: Form-data with image file
**Response**:
```json
{
  "success": true,
  "food_name": "Pizza",
  "confidence": 95.67,
  "calories": 285,
  "serving_size": "1 slice (107g)",
  "nutritional_info": {
    "protein": "12g",
    "carbs": "36g",
    "fat": "10g",
    "fiber": "2.5g",
    "sugar": "4g"
  },
  "top_predictions": [...]
}
```

### 3. Get Prediction History
```
GET /history?limit=10
```
Retrieve past predictions.

### 4. Get Statistics
```
GET /stats
```
Get usage statistics and analytics.

### 5. Delete Prediction
```
DELETE /history/{prediction_id}
```
Delete a specific prediction from history.

## API Documentation

Once the server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Usage Example

### Using cURL

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@pizza.jpg"
```

### Using Python

```python
import requests

url = "http://localhost:8000/predict"
files = {"file": open("pizza.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

### Using JavaScript

```javascript
const formData = new FormData();
formData.append('file', imageFile);

fetch('http://localhost:8000/predict', {
  method: 'POST',
  body: formData
})
.then(response => response.json())
.then(data => console.log(data));
```

## Database Options

### Current: SQLite
- ✅ Simple setup, no configuration
- ✅ Perfect for development
- ❌ Not ideal for high concurrent access

### Recommended for Production: PostgreSQL

1. Install PostgreSQL:
```bash
pip install psycopg2-binary
```

2. Update `database.py` connection:
```python
import psycopg2

def get_connection(self):
    return psycopg2.connect(
        host="localhost",
        database="food_recognition",
        user="your_user",
        password="your_password"
    )
```

### Alternative: MySQL

```bash
pip install mysql-connector-python
```

### Alternative: MongoDB (NoSQL)

```bash
pip install pymongo
```

## Model Training

The project uses MobileNetV2 with transfer learning. To train on custom data:

1. Prepare your dataset in this structure:
```
dataset/
  ├── train/
  │   ├── pizza/
  │   ├── burger/
  │   └── ...
  └── validation/
      ├── pizza/
      ├── burger/
      └── ...
```

2. Create training script:
```python
from food_recognition import FoodRecognitionModel
import tensorflow as tf

# Load datasets
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'dataset/train',
    image_size=(224, 224),
    batch_size=32
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'dataset/validation',
    image_size=(224, 224),
    batch_size=32
)

# Train model
model = FoodRecognitionModel()
model.load_model()
model.train(train_ds, val_ds, epochs=10)
```

## Project Structure

```
GIETPOC/
├── app.py                      # Main FastAPI application
├── food_recognition.py         # Food recognition model
├── calorie_estimator.py        # Calorie estimation logic
├── database.py                 # Database management
├── requirements.txt            # Python dependencies
├── README.md                   # Documentation
├── data/                       # Data directory
│   ├── nutrition_database.json # Nutrition information
│   └── food_recognition.db     # SQLite database
└── models/                     # Model files
    ├── food_model.h5           # Trained model
    └── class_names.json        # Food classes
```

## VSCode Extensions Installed

- ✅ Python (ms-python.python)
- ✅ Pylance (ms-python.vscode-pylance)
- ✅ Python Debugger (ms-python.debugpy)
- ✅ FastAPI Snippets (damildrizzy.fastapi-snippets)

## Development

### Running Tests

```bash
pytest
```

### Code Formatting

```bash
black .
```

### Linting

```bash
flake8
```

## Performance Considerations

- **Image Size**: Larger images take longer to process. Resize before uploading for faster results.
- **Model Loading**: Model loads once at startup. First prediction may be slower.
- **Database**: For production, use PostgreSQL with connection pooling.
- **Caching**: Consider implementing Redis for frequently accessed data.

## Future Enhancements

- [ ] Add user authentication
- [ ] Implement portion size estimation
- [ ] Support for multiple foods in one image
- [ ] Mobile app integration
- [ ] Advanced analytics dashboard
- [ ] Model fine-tuning interface
- [ ] Barcode scanning support
- [ ] Meal planning features

## Contributing

Contributions are welcome! Please feel free to submit pull requests.

## License

MIT License

## Support

For issues or questions, please open an issue on the repository.

## Acknowledgments

- MobileNetV2 architecture from TensorFlow
- FastAPI framework
- Nutrition data from USDA Food Database

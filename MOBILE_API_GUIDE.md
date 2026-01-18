# Mobile API Integration Guide

## 🚀 Quick Start for Mobile Apps

Your Python API now has mobile-optimized endpoints at `/api/mobile/*`

### Base URL

**Local Development:**
```
http://localhost:8000
```

**For Mobile Testing:**
- Android Emulator: `http://10.0.2.2:8000`
- iOS Simulator: `http://localhost:8000`  
- Physical Device: `http://YOUR_COMPUTER_IP:8000` (same WiFi)

---

## 📱 Mobile API Endpoints

All mobile endpoints are under `/api/mobile/` prefix.

### 1. Health Check

```http
GET /api/mobile/health
```

**Response:**
```json
{
  "success": true,
  "status": "online",
  "version": "1.0.0",
  "endpoints": [...]
}
```

### 2. Predict Food (Main Endpoint)

```http
POST /api/mobile/predict
```

**Parameters:**
- `file` (required): Image file
- `save_to_history` (optional): Boolean, default true

**Response:**
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
  "alternatives": [
    {
      "food_name": "Bread",
      "confidence": 2.34
    }
  ],
  "prediction_id": 123
}
```

### 3. Get History

```http
GET /api/mobile/history?limit=20&offset=0
```

**Response:**
```json
{
  "success": true,
  "total_count": 50,
  "items": [
    {
      "id": 123,
      "food_name": "Pizza",
      "confidence": 95.67,
      "calories": 285,
      "timestamp": "2025-12-13 10:30:45"
    }
  ]
}
```

### 4. Get Statistics

```http
GET /api/mobile/stats
```

**Response:**
```json
{
  "success": true,
  "total_predictions": 150,
  "today_predictions": 12,
  "today_calories": 2450,
  "average_confidence": 89.5,
  "most_common_food": "Pizza"
}
```

### 5. Delete Prediction

```http
DELETE /api/mobile/delete/{prediction_id}
```

**Response:**
```json
{
  "success": true,
  "message": "Prediction 123 deleted successfully"
}
```

### 6. Get Available Foods

```http
GET /api/mobile/foods
```

**Response:**
```json
{
  "success": true,
  "total_foods": 25,
  "foods": [
    {
      "name": "Apple",
      "calories": 95,
      "serving_size": "1 medium (182g)",
      "nutritional_info": {
        "protein": "0.5g",
        "carbs": "25g",
        "fat": "0.3g",
        "fiber": "4.4g",
        "sugar": "19g"
      }
    }
  ]
}
```

### 7. Search Foods

```http
GET /api/mobile/search?query=pizza
```

**Response:**
```json
{
  "success": true,
  "query": "pizza",
  "results_count": 1,
  "results": [
    {
      "name": "Pizza",
      "calories": 285,
      "serving_size": "1 slice (107g)",
      "nutritional_info": {
        "protein": "12g",
        "carbs": "36g",
        "fat": "10g",
        "fiber": "2.5g",
        "sugar": "4g"
      }
    }
  ]
}
```

---

## 💻 Integration Examples

### .NET MAUI (C#)

```csharp
// Service class
public class FoodRecognitionService
{
    private readonly HttpClient _httpClient;
    private const string API_BASE_URL = "http://10.0.2.2:8000"; // Android emulator
    
    public FoodRecognitionService()
    {
        _httpClient = new HttpClient { BaseAddress = new Uri(API_BASE_URL) };
    }
    
    public async Task<MobilePredictionResponse> PredictFoodAsync(Stream imageStream, string fileName)
    {
        using var content = new MultipartFormDataContent();
        var imageContent = new StreamContent(imageStream);
        imageContent.Headers.ContentType = new MediaTypeHeaderValue("image/jpeg");
        content.Add(imageContent, "file", fileName);
        
        var response = await _httpClient.PostAsync("/api/mobile/predict", content);
        response.EnsureSuccessStatusCode();
        
        var json = await response.Content.ReadAsStringAsync();
        return JsonSerializer.Deserialize<MobilePredictionResponse>(json);
    }
}

// Response model
public class MobilePredictionResponse
{
    [JsonPropertyName("success")]
    public bool Success { get; set; }
    
    [JsonPropertyName("food_name")]
    public string FoodName { get; set; }
    
    [JsonPropertyName("confidence")]
    public double Confidence { get; set; }
    
    [JsonPropertyName("calories")]
    public int Calories { get; set; }
    
    [JsonPropertyName("serving_size")]
    public string ServingSize { get; set; }
    
    [JsonPropertyName("protein")]
    public string Protein { get; set; }
    
    [JsonPropertyName("carbs")]
    public string Carbs { get; set; }
    
    [JsonPropertyName("fat")]
    public string Fat { get; set; }
    
    [JsonPropertyName("fiber")]
    public string Fiber { get; set; }
    
    [JsonPropertyName("sugar")]
    public string Sugar { get; set; }
    
    [JsonPropertyName("alternatives")]
    public List<Alternative> Alternatives { get; set; }
}

public class Alternative
{
    [JsonPropertyName("food_name")]
    public string FoodName { get; set; }
    
    [JsonPropertyName("confidence")]
    public double Confidence { get; set; }
}
```

### React Native (JavaScript)

```javascript
// API Service
const API_BASE_URL = 'http://10.0.2.2:8000';

export const predictFood = async (imageUri) => {
  const formData = new FormData();
  formData.append('file', {
    uri: imageUri,
    type: 'image/jpeg',
    name: 'food.jpg',
  });

  const response = await fetch(`${API_BASE_URL}/api/mobile/predict`, {
    method: 'POST',
    body: formData,
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });

  return await response.json();
};

export const getHistory = async (limit = 20) => {
  const response = await fetch(
    `${API_BASE_URL}/api/mobile/history?limit=${limit}`
  );
  return await response.json();
};

export const getStats = async () => {
  const response = await fetch(`${API_BASE_URL}/api/mobile/stats`);
  return await response.json();
};
```

### Flutter (Dart)

```dart
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'dart:io';

class FoodRecognitionService {
  static const String baseUrl = 'http://10.0.2.2:8000';
  
  Future<Map<String, dynamic>> predictFood(File imageFile) async {
    var request = http.MultipartRequest(
      'POST',
      Uri.parse('$baseUrl/api/mobile/predict'),
    );
    
    request.files.add(
      await http.MultipartFile.fromPath('file', imageFile.path)
    );
    
    var response = await request.send();
    var responseData = await response.stream.bytesToString();
    return json.decode(responseData);
  }
  
  Future<Map<String, dynamic>> getHistory({int limit = 20}) async {
    final response = await http.get(
      Uri.parse('$baseUrl/api/mobile/history?limit=$limit')
    );
    return json.decode(response.body);
  }
  
  Future<Map<String, dynamic>> getStats() async {
    final response = await http.get(
      Uri.parse('$baseUrl/api/mobile/stats')
    );
    return json.decode(response.body);
  }
}
```

### Swift (iOS)

```swift
import Foundation
import UIKit

class FoodRecognitionService {
    static let baseURL = "http://localhost:8000"
    
    func predictFood(image: UIImage) async throws -> PredictionResponse {
        let url = URL(string: "\(Self.baseURL)/api/mobile/predict")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        
        let boundary = UUID().uuidString
        request.setValue("multipart/form-data; boundary=\(boundary)", 
                        forHTTPHeaderField: "Content-Type")
        
        var body = Data()
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"file\"; filename=\"food.jpg\"\r\n".data(using: .utf8)!)
        body.append("Content-Type: image/jpeg\r\n\r\n".data(using: .utf8)!)
        body.append(image.jpegData(compressionQuality: 0.8)!)
        body.append("\r\n--\(boundary)--\r\n".data(using: .utf8)!)
        
        request.httpBody = body
        
        let (data, _) = try await URLSession.shared.data(for: request)
        return try JSONDecoder().decode(PredictionResponse.self, from: data)
    }
}

struct PredictionResponse: Codable {
    let success: Bool
    let foodName: String
    let confidence: Double
    let calories: Int
    let servingSize: String
    let protein: String
    let carbs: String
    let fat: String
    
    enum CodingKeys: String, CodingKey {
        case success
        case foodName = "food_name"
        case confidence
        case calories
        case servingSize = "serving_size"
        case protein, carbs, fat
    }
}
```

---

## 🧪 Testing the API

### Test Health Endpoint

```bash
curl http://localhost:8000/api/mobile/health
```

### Test Prediction

```bash
curl -X POST "http://localhost:8000/api/mobile/predict" \
  -F "file=@pizza.jpg"
```

### Test History

```bash
curl "http://localhost:8000/api/mobile/history?limit=10"
```

### Test Stats

```bash
curl "http://localhost:8000/api/mobile/stats"
```

---

## 🔧 Configuration

### Find Your Computer's IP

**macOS/Linux:**
```bash
ipconfig getifaddr en0
```

**Windows:**
```cmd
ipconfig
```
Look for "IPv4 Address"

### Update Mobile App

Replace `YOUR_COMPUTER_IP` with your actual IP address in your mobile app code.

---

## 📊 API Features

✅ **Mobile-Optimized**
- Flattened JSON responses
- Pagination support
- Simplified data structures
- Fast response times

✅ **Complete Functionality**
- Food prediction with confidence
- Nutrition information
- History tracking
- Statistics
- Search functionality

✅ **Easy Integration**
- RESTful API
- Standard HTTP methods
- JSON responses
- Clear error messages

---

## 🚀 Next Steps

1. Start the API: `python app.py`
2. Test endpoints: Visit http://localhost:8000/docs
3. Integrate in your MAUI app
4. Test on emulator/device
5. Deploy to production server

---

## 📝 Notes

- All responses include `success: true/false`
- Errors return standard HTTP status codes
- API documentation available at `/docs`
- Mobile endpoints are under `/api/mobile/`

Your API is now ready for mobile integration! 🎉

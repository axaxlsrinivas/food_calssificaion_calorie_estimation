# Food Recognition Model Training Guide

Complete guide to train your own food recognition model on the Food-101 dataset using Google Colab's free GPU.

## 📋 Overview

- **Dataset**: Food-101 (101,000 images, 101 food categories)
- **Training Time**: 2-4 hours with GPU
- **Final Accuracy**: 75-85% (top-1), 90-95% (top-5)
- **Cost**: FREE (using Google Colab)

## 🚀 Quick Start

### Option 1: Full 101 Classes (Recommended for Production)
- All 101 food categories
- Training time: ~4 hours
- Best accuracy: 80-85%

### Option 2: Subset (25 Popular Foods - Recommended for Testing)
- 25 common foods
- Training time: ~1.5 hours  
- Quick testing: 75-80% accuracy

## 📝 Step-by-Step Instructions

### Step 1: Open Google Colab

1. Go to [Google Colab](https://colab.research.google.com/)
2. Sign in with your Google account
3. Create a new notebook: **File** → **New notebook**

### Step 2: Enable GPU

1. Click **Runtime** → **Change runtime type**
2. Select **Hardware accelerator**: **GPU** (T4 or better)
3. Click **Save**

To verify GPU is enabled, run:
```python
import tensorflow as tf
print("GPU Available:", tf.config.list_physical_devices('GPU'))
```

### Step 3: Copy Training Script

1. Open `train_food_model.py` from your project
2. Copy the ENTIRE file content
3. Paste into a new Colab cell
4. Run the cell (Ctrl+Enter or click ▶)

### Step 4: Configure Training

Before running, you can adjust these settings in the script:

```python
# Line ~150 - Choose subset or full dataset
train_ds, val_ds, class_names = prepare_dataset(
    dataset_dir, 
    use_subset=True,    # Change to False for all 101 classes
    num_classes=25      # Ignored if use_subset=False
)
```

**For quick testing (1.5 hours):**
```python
use_subset=True
num_classes=25
```

**For production (4 hours):**
```python
use_subset=False
```

### Step 5: Monitor Training

The script will:
1. ✅ Download Food-101 dataset (~10-20 min)
2. ✅ Prepare training data (~5 min)
3. ✅ Build model (~30 sec)
4. ✅ **Phase 1**: Train classification head (~30 min)
5. ✅ **Phase 2**: Fine-tune entire model (~1-2 hours)
6. ✅ Evaluate and save results

**Expected Output:**
```
============================================================
TRAINING PHASE 1: Classification Head
============================================================
Epoch 1/10
1875/1875 [==============================] - 45s 24ms/step
Validation Accuracy: 0.6234

============================================================
TRAINING PHASE 2: Fine-tuning
============================================================
Epoch 1/10
1875/1875 [==============================] - 120s 64ms/step
Validation Accuracy: 0.7856

============================================================
FINAL RESULTS
============================================================
Validation Accuracy: 0.8123 (81.23%)
Top-5 Accuracy: 0.9456 (94.56%)
============================================================
```

### Step 6: Download Trained Model

After training completes, download these files from Colab:

**Method 1 - Using Files Panel:**
1. Click the **Files** icon (📁) in left sidebar
2. Right-click on each file → **Download**
   - `food_model_trained.h5` (~30MB)
   - `class_names.json` (~2KB)
   - `training_history.png` (optional)
   - `sample_predictions.png` (optional)

**Method 2 - Using Code:**
Add this cell in Colab:
```python
from google.colab import files
files.download('food_model_trained.h5')
files.download('class_names.json')
files.download('training_history.png')
```

### Step 7: Install in Your Project

1. **Copy model file:**
   ```bash
   cp food_model_trained.h5 /Users/cr63494/Documents/GIETPOC/models/food_model.h5
   ```

2. **Copy class names:**
   ```bash
   cp class_names.json /Users/cr63494/Documents/GIETPOC/models/class_names.json
   ```

3. **Update food_recognition.py** to disable Clarifai (optional):
   ```python
   def __init__(self, model_path: str = "models/food_model.h5", use_clarifai: bool = False):
   ```

4. **Restart your API:**
   ```bash
   cd /Users/cr63494/Documents/GIETPOC
   source .venv/bin/activate
   python app.py
   ```

5. **Test predictions:**
   - Upload images in Postman or web interface
   - You should now see 75-85% accuracy with your trained model!

## 🎯 Expected Performance

### Subset Training (25 Foods)
- **Training Time**: 1.5 hours
- **Accuracy**: 75-80%
- **Top-5 Accuracy**: 90-95%
- **Best For**: Quick testing, personal projects

### Full Training (101 Foods)
- **Training Time**: 3-4 hours
- **Accuracy**: 80-85%
- **Top-5 Accuracy**: 93-97%
- **Best For**: Production deployment

## 🔧 Troubleshooting

### Issue: Out of Memory Error
**Solution**: Reduce batch size
```python
batch_size = 16  # Change from 32 to 16
```

### Issue: Training Too Slow
**Solution**: 
1. Verify GPU is enabled (see Step 2)
2. Use subset instead of full dataset
3. Reduce epochs:
```python
history1 = train_phase1(model, train_ds, val_ds, epochs=5)  # Instead of 10
history2 = train_phase2(model, base_model, train_ds, val_ds, epochs=5)
```

### Issue: Accuracy Too Low
**Solutions**:
1. Train for more epochs
2. Use full dataset (101 classes)
3. Adjust learning rate in Phase 2:
```python
optimizer=keras.optimizers.Adam(learning_rate=5e-6)  # Lower learning rate
```

### Issue: Colab Disconnects
**Solution**: 
- Colab free tier has 12-hour limit
- Keep the tab active
- Run this to prevent timeout:
```python
# Add at the beginning
from IPython.display import Javascript
display(Javascript('''
    function KeepClicking(){
        console.log("Clicking");
        document.querySelector("colab-connect-button").click()
    }
    setInterval(KeepClicking, 60000)
'''))
```

## 📊 Understanding the Results

### Training History Plot
- **Blue line**: Training accuracy/loss
- **Orange line**: Validation accuracy/loss
- **Red dashed line**: Where fine-tuning begins
- **Goal**: Validation curves should closely follow training curves

### Sample Predictions
- **Green titles**: Correct predictions
- **Red titles**: Wrong predictions
- **Confidence**: Higher is better (aim for >80%)

## 🎓 Advanced Tips

### 1. Data Augmentation
Increase augmentation for better generalization:
```python
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),      # Increase from 0.1
    layers.RandomZoom(0.2),           # Increase from 0.1
    layers.RandomContrast(0.2),       # Increase from 0.1
    layers.RandomBrightness(0.2),     # Add brightness
])
```

### 2. Different Base Models
Try other architectures:
```python
# Instead of MobileNetV2, use:
base_model = keras.applications.EfficientNetB0(...)  # Better accuracy
base_model = keras.applications.ResNet50(...)        # Classic choice
base_model = keras.applications.InceptionV3(...)     # Good for food
```

### 3. Class Weights
Handle imbalanced classes:
```python
from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight(
    'balanced', classes=np.unique(y_train), y=y_train
)
model.fit(..., class_weight=dict(enumerate(class_weights)))
```

### 4. Save Checkpoints
Save multiple checkpoints during training:
```python
checkpoint = keras.callbacks.ModelCheckpoint(
    'model_epoch_{epoch:02d}_acc_{val_accuracy:.4f}.h5',
    save_freq='epoch'
)
```

## 🌟 Next Steps After Training

1. **Test with real images** - Upload various food images
2. **Analyze errors** - Check which foods are commonly confused
3. **Expand dataset** - Add custom food categories
4. **Deploy to production** - Use your trained model instead of Clarifai
5. **Continuous improvement** - Retrain with user feedback

## 📚 Resources

- **Food-101 Paper**: [https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)
- **Transfer Learning Guide**: [https://www.tensorflow.org/tutorials/images/transfer_learning](https://www.tensorflow.org/tutorials/images/transfer_learning)
- **Google Colab**: [https://colab.research.google.com/](https://colab.research.google.com/)
- **MobileNetV2 Paper**: [https://arxiv.org/abs/1801.04381](https://arxiv.org/abs/1801.04381)

## ⚡ Quick Commands Reference

```bash
# In your local project:

# Activate environment
source .venv/bin/activate

# Start API with trained model
python app.py

# Test predictions
python test_client.py images/pizza.png

# Clear database
python clear_database.py

# Check errors
tail -f logs/app.log
```

## 🎉 Success Checklist

- ✅ GPU enabled in Colab
- ✅ Dataset downloaded and extracted
- ✅ Phase 1 training completed
- ✅ Phase 2 fine-tuning completed
- ✅ Validation accuracy >75%
- ✅ Model and class names downloaded
- ✅ Files copied to project
- ✅ API restarted successfully
- ✅ Test predictions working with high confidence

---

**Need Help?** Common issues and solutions are in the Troubleshooting section above.

**Questions?** Check the Food-101 paper or TensorFlow transfer learning tutorial.

Good luck with training! 🚀

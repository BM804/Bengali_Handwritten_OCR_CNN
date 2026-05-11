# 🔤 Bangla Character Predictor

A deep learning project for recognizing handwritten Bangla (Bengali) characters using a custom Convolutional Neural Network (CNN) trained on the BanglaLekha-Isolated dataset.

---

## 📌 Project Overview

This project builds and trains a CNN model to classify handwritten Bangla characters including numerals, basic characters (vowels & consonants), and compound characters (যুক্তাক্ষর). The model achieves **~95% validation accuracy** across **84 character classes**.

---

## 📥 Download Trained Model

[![Download Model](https://img.shields.io/badge/⬇️_Download-Model_v1.0-blue?style=for-the-badge)](https://github.com/BM804/Bengali_Handwritten_OCR_CNN/releases/download/v1.0/final_bangla_cnn_model.keras)
[![Release](https://img.shields.io/github/v/release/BM804/Bengali_Handwritten_OCR_CNN?style=for-the-badge&color=green)](https://github.com/BM804/Bengali_Handwritten_OCR_CNN/releases/tag/v1.0)

---

## 💻 Project in VS Code

![VS Code](https://raw.githubusercontent.com/BM804/Bengali_Handwritten_OCR_CNN/main/assets/vscode.png)

---

## 📂 Repository Structure

```
Bengali_Handwritten_OCR_CNN/
│
├── assets/                               # Images for README
├── test_images/                          # Sample test images
├── predict.py                            # Inference script
└── README.md
```

> 💡 The trained model is available via [GitHub Releases](https://github.com/BM804/Bengali_Handwritten_OCR_CNN/releases/tag/v1.0) — not stored in the repo.

---

## 📊 Dataset Description

![Dataset Description](https://raw.githubusercontent.com/BM804/Bengali_Handwritten_OCR_CNN/main/assets/dataset.png)

| Property | Details |
|---|---|
| **Type** | Handwritten character dataset |
| **Total Images** | 166,105+ images |
| **Total Classes** | 84 classes |
| **Image Format** | PNG, Grayscale |
| **Image Size** | 32 × 32 pixels |
| **Samples per Class** | ~2,000 |

### Character Categories

| Category | Description |
|---|---|
| **Numbers** | ০, ১, ২, ৩, ৪, ৫, ৬, ৭, ৮, ৯ |
| **Characters** | 50 basic Bangla characters (vowels and consonants) |
| **Compound Characters** | 24 selected যুক্তাক্ষর (conjuncts) |

### Dataset Split

| Split | Percentage |
|---|---|
| Training | 70% |
| Validation | 10% |
| Test | 20% |

---

## 🧠 Model Architecture — `BanglaLekha_CNN`

```
Input: (32, 32, 1) — Grayscale

BLOCK 1:  Conv2D(32) → BN → ReLU → Conv2D(32) → BN → ReLU → MaxPool → Dropout(0.25)
BLOCK 2:  Conv2D(64) → BN → ReLU → Conv2D(64) → BN → ReLU → MaxPool → Dropout(0.25)
BLOCK 3:  Conv2D(128) → BN → ReLU → Conv2D(128) → BN → ReLU → MaxPool → Dropout(0.40)

Flatten → Dense(512, L2) → BN → ReLU → Dropout(0.50)
         → Dense(256, L2) → BN → ReLU → Dropout(0.40)
         → Dense(84, Softmax)
```

### Training Configuration

| Hyperparameter | Value |
|---|---|
| Image Size | 32 × 32 |
| Batch Size | 128 |
| Epochs | 30 |
| Learning Rate | 1e-3 |
| Optimizer | Adam |
| Loss | Categorical Crossentropy |

### Callbacks

- **EarlyStopping** — patience=5, monitors `val_loss`
- **ModelCheckpoint** — saves best model on `val_accuracy`
- **ReduceLROnPlateau** — factor=0.5, patience=2

### Data Augmentation (Training Only)

| Augmentation | Value |
|---|---|
| Rotation | ±10° |
| Width/Height Shift | 10% |
| Zoom | 10% |
| Shear | 10% |
| Fill Mode | Nearest |

---

## 📈 Training Results

### Accuracy Curve

The model reaches **~95% validation accuracy** by epoch 5 and maintains it steadily throughout training.

![Model Accuracy](https://raw.githubusercontent.com/BM804/Bengali_Handwritten_OCR_CNN/main/assets/accuracy.png)

### Loss Curve

Training loss decreases from ~2.5 to ~0.5. Validation loss converges to ~0.2.

![Model Loss](https://raw.githubusercontent.com/BM804/Bengali_Handwritten_OCR_CNN/main/assets/loss.png)

### Confusion Matrix

Strong diagonal dominance across all 84 classes with minimal inter-class confusion.

![Confusion Matrix](https://raw.githubusercontent.com/BM804/Bengali_Handwritten_OCR_CNN/main/assets/confusion_matrix.png)

---

## 🔍 Sample Predictions

![Prediction Output](https://raw.githubusercontent.com/BM804/Bengali_Handwritten_OCR_CNN/main/assets/prediction_output.png)

> Most characters classified with **99%+ confidence**.

---

## ⚙️ Setup & Installation

```bash
pip install tensorflow numpy pandas matplotlib pillow
```

## 🚀 Usage

### Step 1 — Download the model
```bash
# From GitHub Releases
wget https://github.com/BM804/Bengali_Handwritten_OCR_CNN/releases/download/v1.0/final_bangla_cnn_model.keras
```

### Step 2 — Run prediction
```bash
python predict.py
```

Place test images in `test_images/` folder. The script loads the model, preprocesses each image, and displays predictions in a grid.

### Step 3 — Use in your own code
```python
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("final_bangla_cnn_model.keras")

# Preprocess image
img = Image.open("your_image.png").convert("L").resize((32, 32))
img_array = np.array(img) / 255.0
img_array = img_array.reshape(1, 32, 32, 1)

# Predict
prediction = model.predict(img_array)
predicted_class = np.argmax(prediction)
confidence = np.max(prediction) * 100
print(f"Class: {predicted_class}, Confidence: {confidence:.2f}%")
```

---

## 📉 Model Performance Summary

| Metric | Value |
|---|---|
| Test Accuracy | ~90% |
| Validation Accuracy | ~95% |
| Number of Classes | 84 |
| Top-5 Accuracy | High |

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-red?style=flat-square&logo=keras)
![NumPy](https://img.shields.io/badge/NumPy-blue?style=flat-square&logo=numpy)
![Kaggle](https://img.shields.io/badge/Trained_on-Kaggle-20BEFF?style=flat-square&logo=kaggle)

---

## 👤 Author

**BM804** — [@BM804](https://github.com/BM804)

---

## 📄 License

This project is open-source. Feel free to use, modify, and distribute with attribution.

---

## 🙏 Acknowledgements

- Dataset: **BanglaLekha-Isolated**
- Training: [Kaggle](https://www.kaggle.com)
- Framework: [TensorFlow](https://www.tensorflow.org)

# 🔤 Bangla Character Predictor

A deep learning project for recognizing handwritten Bangla (Bengali) characters using a custom Convolutional Neural Network (CNN) trained on the BanglaLekha-Isolated dataset.

---

## 📌 Project Overview

This project builds and trains a CNN model to classify handwritten Bangla characters including numerals, basic characters (vowels & consonants), and compound characters (যুক্তাক্ষর). The model achieves **~95% validation accuracy** across **84 character classes**.

---

## 💻 Project in VS Code

![VS Code](https://raw.githubusercontent.com/BM804/Bengali_Handwritten_OCR_CNN/main/assets/vscode.png)

---

## 📂 Repository Structure

```
Bengali_Handwritten_OCR_CNN/
│
├── model/
│   └── final_bangla_cnn_model.keras     # Trained CNN model
│
├── test_images/                          # Sample test images
│   ├── 01_0001_0_11_0916_1913_63.png
│   └── ...
│
├── assets/                               # Images for README
│   ├── accuracy.png
│   ├── loss.png
│   ├── confusion_matrix.png
│   ├── dataset.png
│   ├── prediction_output.png
│   └── vscode.png
│
├── predict.py                            # Inference script
└── README.md
```

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

```bash
python predict.py
```

Place test images in `test_images/` folder. The script loads the model, preprocesses each image, and displays predictions in a grid.

---

## 📉 Model Performance Summary

| Metric | Value |
|---|---|
| Test Accuracy | ~90% |
| Validation Accuracy | ~95% |
| Number of Classes | 84 |

---

## 🛠️ Tech Stack

`Python` · `TensorFlow/Keras` · `NumPy` · `Matplotlib` · `Pillow`

---

## 👤 Author

**BM804** — [@BM804](https://github.com/BM804)

---

## 🙏 Acknowledgements

- Dataset: **BanglaLekha-Isolated**
- Training: [Kaggle](https://www.kaggle.com)
- Framework: [TensorFlow](https://www.tensorflow.org)

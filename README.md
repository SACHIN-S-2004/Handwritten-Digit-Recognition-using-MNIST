<div align="center">

# ✍️ Handwritten Digit Recognition

### 🧠 CNN-Powered Digit Classifier built with Flask

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Flask-Web_App-black?style=for-the-badge&logo=flask)
![TensorFlow](https://img.shields.io/badge/TensorFlow-CNN_Model-orange?style=for-the-badge&logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-Deep_Learning-red?style=for-the-badge&logo=keras)
![Bootstrap](https://img.shields.io/badge/UI-Bootstrap-purple?style=for-the-badge&logo=bootstrap)

</p>
</div>

---

## ✨ Overview

**Handwritten Digit Recognition** is a Flask web app that identifies handwritten digits (0–9) in real time using a **Convolutional Neural Network (CNN)** trained on the MNIST dataset.

The model achieves a **test accuracy of 99.26%** and supports two input modes — draw directly on a canvas or upload an image — making digit prediction fast and interactive.

📦 From a simple **Jupyter Notebook experiment**, this project has been upgraded into a **fully interactive web application** with:

- ✔ Draw on canvas
- ✔ Upload image
- ✔ Instant prediction
- ✔ Confidence display

All in seconds.

---

## 🎯 Demo Flow

```
Draw digit on canvas / Upload image
            ↓
Preprocess: Grayscale → Resize (28×28) → Normalize
            ↓
Feed into CNN model
            ↓
Predict digit (0–9) + Confidence score
            ↓
Display result on screen
```

---

## 📸 Screenshots

### 💻 Interface

![Interface](sampleScreenshots/Screenshot%20(1908).png)

### 🎨 Canvas Prediction

![Canvas Prediction](sampleScreenshots/Screenshot%20(1909).png)

### 📂 Upload Prediction

![Upload Prediction](sampleScreenshots/Screenshot%20(1910).png)

---

## 🔥 Features

### 🖼️ Input Modes

* Freehand drawing on an HTML5 canvas
* Image file upload for prediction
* Both modes return digit + confidence

### 🧠 CNN Model

* Trained on MNIST (60,000 training images)
* 2× Conv2D + BatchNormalization + MaxPooling + Dropout
* Dense output layer with Softmax (10 classes)
* **Test Accuracy: 99.26%**

### 📊 Prediction Output

* Predicted digit (0–9)
* Confidence percentage
* Instant response via Flask API

### 💎 UI/UX

* Glassmorphism design
* Gradient animated background
* Smooth animations
* Mobile responsive
* Poppins font + Bootstrap powered

### ⚡ Backend

* Flask routing
* Base64 canvas image decoding
* File upload handling
* Fast NumPy + PIL preprocessing

---

## 🧠 How It Works (Simple)

### Step 1 — Preprocess the input

```
Input image
    ↓
Convert to Grayscale
    ↓
Resize to (28 × 28)
    ↓
Normalize pixel values to [0, 1]
    ↓
Reshape to (1, 28, 28, 1)
```

---

### Step 2 — CNN forward pass

```
Conv2D (32 filters) → BatchNorm → MaxPooling → Dropout
        ↓
Conv2D (64 filters) → BatchNorm → MaxPooling → Dropout
        ↓
Flatten → Dense (128) → BatchNorm → Dropout
        ↓
Dense (10) → Softmax
```

---

### Step 3 — Output prediction

```
argmax(predictions)  →  Predicted digit (0–9)
max(predictions)     →  Confidence score (%)
```

---

## 🏗️ Tech Stack

| Layer            | Tech                      |
| ---------------- | ------------------------- |
| Backend          | Flask                     |
| Deep Learning    | TensorFlow / Keras (CNN)  |
| Image Processing | Pillow (PIL)              |
| Math             | NumPy                     |
| Frontend         | HTML + Bootstrap + CSS    |
| Font             | Google Fonts (Poppins)    |

---

## 📂 Project Structure

```
Handwritten-Digit-Recognition-using-MNIST/
│
├── app.py
├── requirements.txt
│
├── model/
│   ├── model.py
│   ├── predict.py
│   └── mnist_cnn_model.keras
│
├── notebooks/
│   ├── digit-recognition-ANN.ipynb
│   └── digit-recognition-CNN.ipynb
│
├── templates/
│   └── index.html
│
├── sampleScreenshots/
│
└── README.md
```

---

## ⚙️ Installation

### 1️⃣ Clone repo

```bash
git clone https://github.com/SACHIN-S-2004/Handwritten-Digit-Recognition-using-MNIST.git
cd Handwritten-Digit-Recognition-using-MNIST
```

---

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

### 3️⃣ Run app

```bash
python app.py
```

---

### 4️⃣ Open browser

```
http://127.0.0.1:5000
```

---

## 📈 Model Performance

| Metric           | Value     |
| ---------------- | --------- |
| Training Samples | 60,000    |
| Test Samples     | 10,000    |
| Test Accuracy    | 99.26%    |
| Optimizer        | Adam      |
| Loss Function    | Categorical Crossentropy |
| Epochs (max)     | 20 (early stopping) |

---

## 🎓 Learning Outcomes

This project demonstrates:

- ✔ Supervised Learning (CNN Classification)
- ✔ Image preprocessing fundamentals
- ✔ Flask backend development
- ✔ Practical Deep Learning deployment

---

## ⭐ If you like this project

Give it a star — it helps a lot!

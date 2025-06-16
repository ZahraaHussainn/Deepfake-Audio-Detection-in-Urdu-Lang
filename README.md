# 🛡️ DeepFake Audio Detection in Urdu – Final Year Project

This project aims to detect AI-generated (deepfake) audio in the **Urdu language**, which remains largely underserved in speech security research. With the rise of synthetic voice cloning technologies like Tacotron and VITS, this system empowers users to identify whether a given voice sample is **real or fake**, helping combat voice-based fraud and misinformation.

The solution combines classical machine learning, deep learning, audio signal processing, and real-time deployment into a **full-stack pipeline**.

---

## 🚀 Project Overview

### 🎯 Objective
Develop a **real-time web application** capable of detecting whether an Urdu audio sample is real (Bonafide) or fake (Spoofed using Tacotron or VITS TTS).

---

## 🧪 Key Components

### 1. 📁 Dataset Handling
- Urdu dataset containing audio clips from:
  - ✅ **Bonafide speakers**
  - 🎭 **Spoofed Tacotron-generated samples**
  - 🎭 **Spoofed VITS-generated samples**
- **Speaker-disjoint splitting** to avoid data leakage.

### 2. 🔊 Audio Preprocessing
- Audio operations: resampling, trimming, normalization.
- Feature extraction using:
  - MFCCs (Mel Frequency Cepstral Coefficients)
  - Mel Spectrograms
  - HuBERT Embeddings (via Hugging Face models)

### 3. 🧠 Model Training & Evaluation

#### 🔹 Classical Machine Learning Models
- SVM (Support Vector Machine)
- Decision Tree
- Random Forest

#### 🔹 Deep Learning Models
- MLP (Multi-Layer Perceptron)
- MLP + Additive Attention (focus on important frames)
- CNN (Convolutional Neural Network)
- ViT (Vision Transformer) – trained on Mel Spectrogram images

✅ All models were evaluated using **speaker-disjoint validation** with performance metrics like:
- Accuracy
- Precision
- Recall
- F1-score

---

### 4. 🔬 Experiments & Results
- Comparative analysis of all models
- **ViT and MLP+Attention** performed best
- Experiment notebooks are available in the [`/notebooks/`](notebooks/) directory

---

## 🌐 Web App Deployment

### 🔧 Built With
- **Flask** (Python backend) – handles audio upload and inference
- **HTML / CSS / Bootstrap** – for responsive UI
- **MySQL** – optional, to log predictions and user feedback

💡 **Users upload an audio file through the browser and get an instant result**: **Real** or **Fake**

---

## 🧰 Tech Stack

### 🔹 Languages & Frameworks
- Python
- Flask
- MySQL
- HTML / CSS

### 🔹 Libraries
- PyTorch
- Librosa
- Scikit-learn
- Hugging Face Transformers
- NumPy, Pandas, Matplotlib

### 🔹 Models
- SVM, Decision Tree, Random Forest
- MLP, MLP + Additive Attention
- CNN
- Vision Transformer (ViT)

### 🔹 Tools
- Google Colab
- Jupyter Notebooks
- Git
  

---


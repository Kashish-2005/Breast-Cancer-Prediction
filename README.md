🩺 Breast Cancer Prediction System

An AI-powered web application that predicts whether a breast tumor is Benign or Malignant using machine learning.
The system is built using TensorFlow (Keras) for model training and Streamlit for deployment.

🚀 Live Overview

This project demonstrates a complete ML pipeline:

Data preprocessing & scaling

Neural network training

Model saving & reuse

Web-based inference using Streamlit

📊 Dataset

Dataset: Breast Cancer Wisconsin (Diagnostic) Dataset

Source: UCI Machine Learning Repository / sklearn datasets

Features: 30 numerical features describing tumor characteristics

Target:

0 → Benign

1 → Malignant

🧠 Model Details

Algorithm: Neural Network (Keras / TensorFlow)

Input Features: 30 (Mean, Standard Error, Worst values)

Loss Function: Binary Crossentropy

Optimizer: Adam

Evaluation Metric: Accuracy

Preprocessing: StandardScaler (saved and reused during inference)

🖥️ Web Application (Streamlit)
Features:

User-friendly UI with categorized inputs

Manual numeric input for all 30 features

Real-time prediction

Confidence score display

Medical disclaimer included

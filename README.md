# 🌱 Plant Disease Detection & AI Report

## 📌 Project Overview
This project is a **deep learning-based application** for detecting plant diseases using images of leaves. It leverages a **pre-trained VGG16 model** fine-tuned for classification and integrates AI-powered report generation using **Groq’s Mixtral-8x7b** model.

## 🚀 Features
- **Upload plant leaf images** for disease classification.
- **Predict disease** with a trained deep learning model.
- **Generate an AI-driven disease report** with a description, symptoms, causes, and treatments.
- **User-friendly web app** built with `Streamlit`.

## 🔍 Problem Statement
Farmers and agriculturists struggle with identifying plant diseases early, which can lead to reduced crop yield and economic losses. This project aims to provide an **automated, AI-driven solution** for detecting diseases in plants and offering actionable insights.

## 🏗️ Tech Stack
- **Deep Learning:** TensorFlow & Keras (VGG16 model)
- **Web Framework:** Streamlit
- **AI Report Generation:** Groq API (Mixtral-8x7b model)
- **Libraries Used:** NumPy, PIL, Matplotlib, TensorFlow, Groq

## 🛠️ Installation & Setup
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/ldotmithu/Smart-Agriculture-Disease-Detection-with-AI-Driven-Solutions.git]
   cd Smart-Agriculture-Disease-Detection-with-AI-Driven-Solutions
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Set up API keys:**
   - Create a `.env` file.
   - Add your Groq API key:
     ```env
     groq_api_key = "your_groq_api_key"
     ```
4. **Run the application:**
   ```bash
   streamlit run app.py
   ```

## 📊 Model Details
- **Base Model:** VGG16 (pre-trained on ImageNet)
- **Fine-Tuned for:** Plant disease classification
- **Input Size:** 224x224 RGB images
- **Classes:** Pepper and Tomato disease types

## 📸 Example Usage
1. **Upload a plant leaf image** via the Streamlit interface.
2. **Model predicts the disease** and provides a confidence score.
3. **AI generates a professional disease report** with insights and solutions.

## 📝 Future Improvements
- Expand to more plant species.
- Improve AI-generated reports with more granular data.
- Deploy as a cloud-based API service.

## 📜 License
This project is licensed under the **MIT License**.

## Demo 
<!-- Failed to upload "Screen Recording 2025-03-14 at 13.07.28.mov" -->

---
🌍 **Empowering farmers with AI-driven plant health insights!**


import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import groq
import matplotlib.pyplot as plt
from PIL import Image


MODEL_PATH = "plant_disease_model_final.h5"  
model = tf.keras.models.load_model('final_plant_model.h5')

class_names = ['Pepper__bell___Bacterial_spot',
 'Pepper__bell___healthy',
 'Tomato_Early_blight',
 'Tomato_Late_blight',
 'Tomato_healthy']


client = groq.Client(api_key="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")  

def predict_disease(img):
    img = img.resize((224, 224))  
    img_array = image.img_to_array(img) / 255.0  
    img_array = np.expand_dims(img_array, axis=0)  

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]  
    return predicted_class


def generate_disease_report(disease):
    prompt = f"""
    Generate a professional agricultural disease report for a plant diagnosed with {disease} .
    Include:
    - Description of the disease
    - Symptoms and effects on crops
    - Possible causes
    - Recommended treatments and prevention strategies
    give me sort usefull answer
    """

    response = client.chat.completions.create(
        model="mixtral-8x7b-32768", 
        messages=[{"role": "system", "content": prompt}]
    )

    return response.choices[0].message.content


st.set_page_config(page_title="Plant Disease Detector", layout="centered")

st.title("🌱 Plant Disease Detection & AI Report")
st.write("Upload an image of a plant leaf, and the AI will predict the disease and generate a medical report.")


uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

   
    predicted_disease = predict_disease(img)
    st.success(f"**Prediction: {predicted_disease}**")

    
    with st.spinner("Generating AI report..."):
        report = generate_disease_report(predicted_disease)

    st.subheader("📝 AI-Generated Disease Report")
    st.write(report)

st.markdown("---")
st.markdown("🔍 Developed with **VGG16 + Groq AI**")

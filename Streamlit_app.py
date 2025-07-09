import streamlit as st
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# 🔹 Load the model (TFLite)
interpreter = tf.lite.Interpreter(model_path="plant_disease_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 🔹 Load class labels from label.txt
with open("label.txt", "r") as f:
    class_labels = [line.strip() for line in f.readlines()]

# 🔹 Load medicine data from medicine.json
with open("medicine.json", "r", encoding="utf-8") as f:
    all_medicines = json.load(f)

# 🔹 Language selector
language = st.selectbox("🌐 Choose Language / भाषा चुनें", ["English", "Hindi"])

# 🔹 UI headings
if language == "English":
    st.title("🌿 AI Farmer Helper")
    st.markdown("Upload a crop leaf image to detect disease and get treatment.")
else:
    st.title("🌿 एआई किसान सहायक")
    st.markdown("फसल की पत्ती की तस्वीर अपलोड करें और बीमारी की पहचान करें।")

# 🔹 File uploader
uploaded_file = st.file_uploader("📷 Upload Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="📸", use_column_width=True)

    if st.button("🔍 Detect Disease" if language == "English" else "🔍 बीमारी की पहचान करें"):
        # Load and preprocess image
        img = image.load_img(uploaded_file, target_size=(128, 128))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # 🔹 TFLite Prediction
        interpreter.set_tensor(input_details[0]['index'], img_array.astype(np.float32))
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])
        class_index = np.argmax(prediction[0])
        disease = class_labels[class_index]
        confidence = round(np.max(prediction[0]) * 100, 2)

        # 🔹 Get medicine based on selected language
        lang_key = "hi" if language == "Hindi" else "medicine"
        medicine = all_medicines.get(disease, {}).get(lang_key, "No treatment info available." if language == "English" else "इलाज की जानकारी उपलब्ध नहीं है।")

        # 🔹 Output
        st.success(f"🦠 Disease: {disease}" if language == "English" else f"🦠 बीमारी: {disease}")
        st.info(f"💊 Treatment: {medicine}" if language == "English" else f"💊 उपचार: {medicine}")
        st.caption(f"🔎 Confidence: {confidence}%")

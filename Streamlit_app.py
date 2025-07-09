import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# 🔹 Load TFLite model
interpreter = tf.lite.Interpreter(model_path="plant_disease_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class_labels = ['Pepper___Bacterial_spot', 'Tomato___Healthy', 'Tomato___Late_blight']

medicine_dict = {
    "English": {
        "Tomato___Late_blight": "Spray with fungicides like Captan or Mancozeb.",
        "Tomato___Healthy": "The plant is healthy. No medicine required.",
        "Pepper___Bacterial_spot": "Spray with copper spray or streptomycin."
    },
    "Hindi": {
        "Tomato___Late_blight": "कप्तान या मैनकोजेब जैसे फफूंदनाशकों का छिड़काव करें।",
        "Tomato___Healthy": "पौधा स्वास्थ है, किसी दवा की आवश्यक्ता नहीं है।",
        "Pepper___Bacterial_spot": "कॉपर स्प्रे या स्ट्रेप्टोमायसिन का छिड़काव करें।"
    }
}

language = st.selectbox("🌐 Choose Language / भाषा चुनें", ["English", "Hindi"])

st.title("🌿 AI Farmer Helper" if language == "English" else "🌿 एआई किसान सहायक")
st.markdown("Upload a crop leaf image to detect disease and get treatment." if language == "English" else "फसल की पत्ती की तस्वीर अपलोड करें और बीमारी की पहचान करें।")

uploaded_file = st.file_uploader("📷 Upload Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="📸", use_column_width=True)

    if st.button("🔍 Detect Disease" if language == "English" else "🔍 बीमारी की पहचान करें"):
        img = Image.open(uploaded_file).resize((128, 128)).convert('RGB')
        img_array = np.expand_dims(np.array(img) / 255.0, axis=0).astype(np.float32)

        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])

        class_index = np.argmax(prediction)
        disease = class_labels[class_index]
        confidence = round(np.max(prediction) * 100, 2)

        medicine = medicine_dict[language].get(disease, "No treatment info available." if language == "English" else "इलाज की जानकारी उपलब्ध नहीं है।")

        st.success(f"🦠 Disease: {disease}" if language == "English" else f"🦠 बीमारी: {disease}")
        st.info(f"💊 Treatment: {medicine}" if language == "English" else f"💊 उपचार: {medicine}")
        st.caption(f"🔎 Confidence: {confidence}%")

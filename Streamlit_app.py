import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# 🔹 Load the model
model = load_model("plant_disease_model.h5")

# ✅ Correct class label order based on training
class_labels = ['Pepper___Bacterial_spot', 'Tomato___Healthy', 'Tomato___Late_blight']  # Ensure this matches training order

# 🔹 Language-based medicine dictionary
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

        # Prediction
        prediction = model.predict(img_array)
        class_index = np.argmax(prediction[0])
        disease = class_labels[class_index]
        confidence = round(np.max(prediction[0]) * 100, 2)

        # Medicine
        medicine = medicine_dict[language].get(
            disease, 
            "No treatment info available." if language == "English" else "इलाज की जानकारी उपलब्ध नहीं है।"
        )

        # Output
        st.success(f"🦠 Disease: {disease}" if language == "English" else f"🦠 बीमारी: {disease}")
        st.info(f"💊 Treatment: {medicine}" if language == "English" else f"💊 उपचार: {medicine}")
        st.caption(f"🔎 Confidence: {confidence}%")

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Set page layout
st.set_page_config(page_title="🌾 AI Farmer Helper", layout="centered")

# Title
st.title("🌿 Plant Disease Detection")

# Load TFLite model
@st.cache_resource
def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path="plant_disease_model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Class labels (change if your model has different classes)
class_labels = ['Pepper___Bacterial_spot', 'Tomato___Healthy', 'Tomato___Late_blight']

# Medicine dictionary
medicine_dict = {
    "Pepper___Bacterial_spot": "बैक्टीरियल स्पॉट के लिए कॉपर आधारित फफूंदनाशक का छिड़काव करें।",
    "Tomato___Healthy": "पौधा स्वस्थ है, कोई दवा आवश्यक नहीं है।",
    "Tomato___Late_blight": "लेट ब्लाइट के लिए मेटालेक्सिल युक्त फफूंदनाशक का उपयोग करें।"
}

# Upload image
uploaded_file = st.file_uploader("🖼️ पत्ते की तस्वीर अपलोड करें", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Preprocess image
        img = Image.open(uploaded_file).convert("RGB")
        img = img.resize((128, 128))  # Make sure this matches model's input
        img_array = np.expand_dims(np.array(img) / 255.0, axis=0).astype(np.float32)

        # Run prediction
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])

        predicted_class = class_labels[np.argmax(prediction)]
        confidence = float(np.max(prediction))

        # Show results
        st.success(f"🔍 पहचानी गई बीमारी: **{predicted_class}**")
        st.info(f"💊 सुझावित दवा: {medicine_dict.get(predicted_class, 'जानकारी उपलब्ध नहीं है।')}")
        st.progress(confidence)

    except Exception as e:
        st.error("❌ पूर्वानुमान में त्रुटि आई। कृपया सही तस्वीर अपलोड करें या मॉडल जांचें।")
        st.exception(e)

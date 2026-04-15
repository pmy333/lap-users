import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open("Model (4).pkl", "rb"))

# Page config
st.set_page_config(page_title="ML App", layout="centered")

# Title
st.title("🚀 Machine Learning Web App")
st.markdown("### Enter input values to get predictions")

# Sidebar
st.sidebar.header("About")
st.sidebar.write("This app uses a trained ML model to make predictions.")

# --- INPUT FIELDS (MODIFY BASED ON YOUR MODEL) ---
st.subheader("Input Features")

feature1 = st.number_input("Feature 1", value=0.0)
feature2 = st.number_input("Feature 2", value=0.0)
feature3 = st.number_input("Feature 3", value=0.0)
feature4 = st.number_input("Feature 4", value=0.0)

# Convert inputs into array
input_data = np.array([[feature1, feature2, feature3, feature4]])

# Prediction button
if st.button("Predict"):
    try:
        prediction = model.predict(input_data)
        
        st.success(f"✅ Prediction: {prediction[0]}")
        
    except Exception as e:
        st.error(f"Error: {e}")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit")

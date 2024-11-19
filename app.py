import streamlit as st
import joblib
import pandas as pd

# Load the pre-trained model and vectorizer
model = joblib.load('fake_news_detector_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Streamlit app title
st.title("Fake News Detection App")

# Input text box for news content
user_input = st.text_area("Enter the news text here:")

# Button to classify news as Fake or Real
if st.button("Classify"):
    if user_input:
        # Preprocess and vectorize the input text
        input_data = vectorizer.transform([user_input.lower()])
        
        # Predict using the loaded model
        prediction = model.predict(input_data)
        
        # Display result
        if prediction[0] == 1:
            st.success("The news is classified as: Real")
        else:
            st.error("The news is classified as: Fake")
    else:
        st.warning("Please enter some text to classify.")

# Optional: Display more model info and metrics if needed
st.sidebar.title("About the App")
st.sidebar.info(
    "This app uses a machine learning model to classify news as Fake or Real. "
    "Enter any news text to get the prediction."
)

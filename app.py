import streamlit as st
import pickle

# Load the vectorizer
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Load the model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Streamlit app
st.set_page_config(page_title="Fake News Detector", layout="centered")

st.title("📰 Fake News Detector")
st.markdown("Enter a news article or headline below to check if it's real or fake.")

# Text input
user_input = st.text_area("Paste your news text here:", height=200)

# Predict button
if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Transform input and predict
        transformed_input = vectorizer.transform([user_input])
        prediction = model.predict(transformed_input)[0]

        # Display result
        if prediction == 1:
            st.success("✅ This news looks **Real**.")
        else:
            st.error("🚨 This news seems **Fake**.")


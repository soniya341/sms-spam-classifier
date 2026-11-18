import streamlit as st
import joblib

model = joblib.load("data/spam_model.pkl")
vec = joblib.load("data/vectorizer.pkl")

st.title("SMS Spam Classifier")
st.write("Type a message and I will tell you if it is spam.")

text = st.text_area("Your message")

if st.button("Check"):
    if text.strip() == "":
        st.warning("Please type a message first.")
    else:
        result = model.predict(vec.transform([text]))[0]
        if result == 1:
            st.error("SPAM")
        else:
            st.success("NOT SPAM")
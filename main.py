# ---------------- IMPORT LIBRARIES ----------------
import streamlit as st
import numpy as np
import tensorflow as tf

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="IMDB Sentiment Analysis",
    page_icon="🎬",
    layout="centered"
)

# ---------------- LOAD DATA ----------------
word_index = imdb.get_word_index()
reverse_word_index = {
    value: key for key, value in word_index.items()
}

# ---------------- LOAD MODEL ----------------
model = load_model('simple_rnn_imdb.h5')

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

.main-title {
    font-size: 40px;
    font-weight: bold;
    text-align: center;
    color: #FF4B4B;
}

.sub-text {
    text-align: center;
    color: gray;
    margin-bottom: 30px;
}

.review-box {
    background-color: #f5f5f5;
    padding: 15px;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HELPER FUNCTIONS ----------------

# Decode encoded review
def decode_review(encoded_review):
    return ' '.join([
        reverse_word_index.get(i - 3, '?')
        for i in encoded_review
    ])

# Preprocess user review
def preprocess_text(text):

    words = text.lower().split()

    encoded_review = [
        word_index.get(word, 2) + 3
        for word in words
    ]

    padded_review = sequence.pad_sequences(
        [encoded_review],
        maxlen=500
    )

    return padded_review

# ---------------- SIDEBAR ----------------
st.sidebar.title("About")

st.sidebar.write("""
This application performs:

Movie Review Sentiment Analysis

Built using:
- TensorFlow
- Embedding Layer
- SimpleRNN
- Streamlit
""")

st.sidebar.markdown("---")

st.sidebar.info(
    "Example Review:\n\n"
    "'This movie was absolutely amazing and inspiring!'"
)

# ---------------- MAIN TITLE ----------------
st.markdown(
    '<p class="main-title">🎬 IMDB Sentiment Analysis</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="sub-text">Analyze whether a movie review is Positive or Negative using Simple RNN</p>',
    unsafe_allow_html=True
)

# ---------------- USER INPUT ----------------
user_input = st.text_area(
    "Enter Movie Review",
    placeholder="Type your movie review here..."
)

# ---------------- PREDICTION ----------------
if st.button("Analyze Sentiment"):

    if user_input.strip() == "":

        st.warning("⚠️ Please enter a movie review.")

    else:

        # Preprocess Input
        preprocessed_input = preprocess_text(user_input)

        # Predict
        with st.spinner("Analyzing review sentiment..."):

            prediction = model.predict(preprocessed_input)

            prediction_score = prediction[0][0]

            sentiment = (
                "Positive 😊"
                if prediction_score > 0.5
                else "Negative 😞"
            )

        # ---------------- DISPLAY RESULTS ----------------
        st.subheader("Prediction Result")

        # Metric Card
        st.metric(
            label="Prediction Score",
            value=f"{prediction_score:.2f}"
        )

        # Progress Bar
        st.progress(float(prediction_score))

        # Sentiment Message
        if prediction_score > 0.5:

            st.success(
                f"✅ Sentiment: {sentiment}"
            )

        else:

            st.error(
                f"❌ Sentiment: {sentiment}"
            )

        # ---------------- SHOW REVIEW ----------------
        with st.expander("View Your Review"):

            st.write(user_input)

# ---------------- FOOTER ----------------
st.markdown("---")

st.write(
    "Built using TensorFlow + SimpleRNN + Streamlit 🚀"
)
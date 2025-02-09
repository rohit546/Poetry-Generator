import pickle
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 🎨 Custom Styling
st.markdown(
    """
    <style>
        /* Background and Text */
        body {
            background-color: #121212;
            color: #E0E0E0;
        }
        /* Title Styling */
        .title {
            font-size: 36px !important;
            font-weight: bold;
            text-align: center;
            color: #FFA500;
        }
        /* Text Input */
        .stTextInput>div>div>input {
            font-size: 18px;
            padding: 10px;
            border-radius: 8px;
            border: 2px solid #FFA500;
        }
        /* Poetry Output */
        .poetry-box {
            background-color: #1E1E2E;
            padding: 15px;
            border-radius: 10px;
            font-size: 20px;
            text-align: center;
            color: #FFD700;
            font-weight: bold;
        }
        /* Custom Button */
        .stButton>button {
            background-color: #FFA500;
            color: white;
            font-size: 18px;
            border-radius: 8px;
            padding: 10px;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #FF8C00;
            transform: scale(1.05);
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ✅ Load Model
model = load_model("poetry_model.keras")

# ✅ Load Tokenizer
with open("tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)

# ✅ Generate Poetry Function
def generate_poetry(seed_text, word_count=10):
    for _ in range(word_count):
        sequence = tokenizer.texts_to_sequences([seed_text])[0]
        sequence = pad_sequences([sequence], maxlen=20, padding='pre')  # Adjust maxlen as needed
        predicted_index = np.argmax(model.predict(sequence), axis=-1)[0]
        predicted_word = tokenizer.index_word.get(predicted_index, "")
        seed_text += " " + predicted_word
    return seed_text

# ✅ Streamlit UI
st.markdown('<h1 class="title">📜 Roman Urdu Poetry Generator</h1>', unsafe_allow_html=True)

# Input Field
seed_text = st.text_input("💬 Enter a word:", "Pyar")

# Word Count Slider
word_count = st.slider("🔢 Number of words to generate:", min_value=5, max_value=50, value=20)

# Generate Button
if st.button("✨ Generate Poetry"):
    poetry = generate_poetry(seed_text, word_count)

    # Poetry Output
    st.markdown('<h3 style="text-align: center;">✨ Generated Poetry:</h3>', unsafe_allow_html=True)
    st.markdown(f'<div class="poetry-box">{poetry}</div>', unsafe_allow_html=True)

    # Copy to Clipboard Feature
    st.code(poetry, language="markdown")

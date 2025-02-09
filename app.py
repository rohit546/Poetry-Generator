import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

# 🎨 Custom Styling
st.markdown(
    """
    <style>
        /* Background Color */
        body {
            background-color: #1E1E2E;
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
        }
        /* Poetry Output */
        .poetry-box {
            background-color: #2E2E3E;
            padding: 15px;
            border-radius: 10px;
            font-size: 20px;
            text-align: center;
        }
        /* Custom Button */
        .stButton>button {
            background-color: #FFA500;
            color: white;
            font-size: 18px;
            border-radius: 8px;
            padding: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<h1 class="title">📜 Roman Urdu Poetry Generator</h1>', unsafe_allow_html=True)

# ✅ Sidebar for Settings
st.sidebar.header("⚙️ Settings")
word_count = st.sidebar.slider("Number of words to generate:", min_value=5, max_value=50, value=20)

# ✅ Check if model file exists
if not os.path.exists("poetry_model.h5"):
    st.error("❌ Model file not found! Make sure poetry_model.h5 is uploaded.")
    st.stop()

# ✅ Load Model with Error Handling
try:
    model = load_model("poetry_model.h5")
    st.sidebar.success("✅ Model Loaded Successfully!")
except Exception as e:
    st.sidebar.error(f"❌ Model loading failed! Error: {str(e)}")
    st.stop()

# ✅ Load Tokenizer with Error Handling
try:
    with open("tokenizer.pkl", "rb") as handle:
        tokenizer = pickle.load(handle)
    st.sidebar.success("✅ Tokenizer Loaded Successfully!")
except Exception as e:
    st.sidebar.error(f"❌ Tokenizer loading failed! Error: {str(e)}")
    st.stop()

# ✅ Poetry Generation Function
def generate_poetry(seed_text, word_count=10):
    seed_text = seed_text.strip().split()[0]  # Ensure only one word is taken
    generated_text = seed_text

    for _ in range(word_count):
        sequence = tokenizer.texts_to_sequences([generated_text])[0]
        if not sequence:
            return "⚠️ Word not found in vocabulary. Try another word!"
        
        sequence = pad_sequences([sequence], maxlen=20, padding='pre')
        predicted_index = np.argmax(model.predict(sequence), axis=-1)[0]
        predicted_word = tokenizer.index_word.get(predicted_index, "")
        if not predicted_word:
            break

        generated_text += " " + predicted_word
    return generated_text

# ✅ Streamlit UI Components
col1, col2 = st.columns([2, 1])

with col1:
    seed_text = st.text_input("💬 Enter a Word:", "").strip()

with col2:
    st.markdown("### ")
    generate_btn = st.button("✨ Generate Poetry")

if generate_btn:
    if not seed_text or " " in seed_text:
        st.error("❌ Please enter only **one** word!")
    else:
        poetry = generate_poetry(seed_text, word_count)

        # ✅ Animated Poetry Display
        with st.container():
            st.markdown('<h3 style="text-align: center;">✨ Generated Poetry:</h3>', unsafe_allow_html=True)
            st.markdown(f'<div class="poetry-box">{poetry}</div>', unsafe_allow_html=True)

        # ✅ Copy to Clipboard Feature
        st.code(poetry, language="markdown")

import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

st.title("📜 Roman Urdu Poetry Generator")

# ✅ Check if model file exists
if not os.path.exists("poetry_model.h5"):
    st.error("❌ Model file not found! Make sure poetry_model.h5 is uploaded.")
    st.stop()

# ✅ Load the Model with Error Handling
try:
    model = load_model("poetry_model.h5")  # Use .h5 format for better compatibility
    st.success("✅ Model Loaded Successfully!")
except Exception as e:
    st.error(f"❌ Model loading failed! Error: {str(e)}")
    st.stop()

# ✅ Load the Tokenizer with Error Handling
try:
    with open("tokenizer.pkl", "rb") as handle:
        tokenizer = pickle.load(handle)
    st.success("✅ Tokenizer Loaded Successfully!")
except Exception as e:
    st.error(f"❌ Tokenizer loading failed! Error: {str(e)}")
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

# ✅ Streamlit UI
seed_text = st.text_input("Enter a word:", "").strip()
word_count = st.slider("Number of words to generate:", min_value=5, max_value=50, value=20)

if st.button("Generate"):
    if not seed_text or " " in seed_text:
        st.error("❌ Please enter only **one** word!")
    else:
        poetry = generate_poetry(seed_text, word_count)
        st.subheader("✨ Generated Poetry:")
        st.write(poetry)

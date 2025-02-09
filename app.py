import pickle
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

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
st.title("Roman Urdu Poetry Generator")
seed_text = st.text_input("Enter a word:", "zindagi")
word_count = st.slider("Number of words to generate:", min_value=5, max_value=50, value=20)

if st.button("Generate"):
    poetry = generate_poetry(seed_text, word_count)
    st.write("**Generated Poetry:**")
    st.write(poetry)

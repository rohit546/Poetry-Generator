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
            break  # Stop if no valid word is predicted

        generated_text += " " + predicted_word

    return generated_text

# ✅ Streamlit UI
st.title("📜 Roman Urdu Poetry Generator")
st.write("Enter a **single** word to generate poetry!")

# ✅ User Input
seed_text = st.text_input("Enter a word:", "").strip()
word_count = st.slider("Number of words to generate:", min_value=5, max_value=50, value=20)

# ✅ Button to generate poetry
if st.button("Generate"):
    if not seed_text or " " in seed_text:
        st.error("❌ Please enter only **one** word!")
    else:
        poetry = generate_poetry(seed_text, word_count)
        st.subheader("✨ Generated Poetry:")
        st.write(poetry)

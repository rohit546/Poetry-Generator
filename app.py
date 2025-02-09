import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ✅ Load the trained model
model = load_model("poetry_model.keras")  # Ensure you saved the model in this format

# ✅ Load tokenizer
with open("tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)

# ✅ Define max sequence length (Make sure it matches training)
max_seq_length = 50  # Update this based on your model

# ✅ Poetry generation function
def generate_poetry(seed_text, word_count=10):
    seed_text = seed_text.strip()
    if not seed_text:
        return "⚠️ Please enter a valid word!"

    generated_text = seed_text
    for _ in range(word_count):
        sequence = tokenizer.texts_to_sequences([generated_text])[0]

        if not sequence:
            return "⚠️ Word not found in vocabulary. Try another word!"

        sequence = pad_sequences([sequence], maxlen=max_seq_length, padding='pre')
        predicted_index = np.argmax(model.predict(sequence), axis=-1)[0]

        predicted_word = tokenizer.index_word.get(predicted_index, "")
        if not predicted_word:
            break  # Stop if no valid word is predicted

        generated_text += " " + predicted_word
    return generated_text

# ✅ Streamlit UI
st.title("📜 Roman Urdu Poetry Generator")
st.write("Enter a word to generate poetry!")

# ✅ User input field
user_input = st.text_input("Enter a word:", "")

# ✅ Button to generate poetry
if st.button("Generate Poetry"):
    output_poetry = generate_poetry(user_input, word_count=20)
    st.subheader("Generated Poetry:")
    st.write(output_poetry)

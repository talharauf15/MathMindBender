import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the model and tokenizer
model_path = "math_riddle_model_final"  # Change if needed
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Streamlit UI
st.title("Math Riddle Generator 🤖")
st.write("Enter a prompt, and the model will generate a math riddle!")

user_input = st.text_input("Enter your prompt:", "")

if st.button("Generate Riddle"):
    if user_input:
        inputs = tokenizer(user_input, return_tensors="pt")
        output = model.generate(**inputs, max_length=100)
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        st.write("**Generated Riddle:**")
        st.success(response)
    else:
        st.warning("Please enter a prompt first!")

# Footer
st.sidebar.markdown("### Model Info")
st.sidebar.text(f"Loaded Model: {model_path}")

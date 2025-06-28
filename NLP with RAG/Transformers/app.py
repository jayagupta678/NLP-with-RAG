import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Title
st.title("Mini GPT Text Generator")

# Load model (DistilGPT2 for lightweight inference)
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    return tokenizer, model

tokenizer, model = load_model()

# User input
prompt = st.text_area("Enter your prompt", "Once upon a time")

# Generate button
if st.button("Generate Text"):
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Inference
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=100,
            num_return_sequences=1,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode output
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    st.subheader(" Generated Text")
    st.write(generated)


# Streamlit run app.py

# pip install streamlit torch transformers

# python == 3.8-3.10
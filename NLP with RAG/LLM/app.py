import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

st.title(" Prompt-Based LLM with Flan-T5 (Small)")

@st.cache_resource
def load_model():
    model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

prompt = st.text_area("Enter your prompt (e.g., Translate to French, Summarize, etc.):", "Summarize: The sun is the center of our solar system...")

if st.button("Generate Response"):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=100,
            do_sample=True,
            temperature=0.9,
            top_p=0.95
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    st.subheader(" Response:")
    st.write(response)

import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import PyPDF2
import uuid

# Setup ChromaDB
client = chromadb.Client(Settings(anonymized_telemetry=False))
collection = client.get_or_create_collection("rag_pdf")

# Load models
@st.cache_resource
def load_models():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return tokenizer, model, embedder

tokenizer, model, embedder = load_models()

# Streamlit UI
st.title(" Chat with Your PDF using RAG")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    all_text = ""

    for page in pdf_reader.pages:
        text = page.extract_text()
        if text:
            all_text += text + "\n"

    # Split into chunks
    chunks = [all_text[i:i+500] for i in range(0, len(all_text), 500)]
    embeddings = embedder.encode(chunks).tolist()
    ids = [str(uuid.uuid4()) for _ in range(len(chunks))]

    # Store in ChromaDB
    collection.add(documents=chunks, embeddings=embeddings, ids=ids)
    st.success(f"{len(chunks)} chunks stored in vector DB.")

    # Ask question
    user_query = st.text_input("❓ Ask a question from the PDF:")

    if user_query:
        query_embedding = embedder.encode([user_query])[0].tolist()
        results = collection.query(query_embeddings=[query_embedding], n_results=3)
        context = " ".join(results["documents"][0])

        # Form final prompt
        prompt = f"Context: {context}\n\nQuestion: {user_query}\n\nAnswer:"

        # Generate answer
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        output_ids = model.generate(input_ids, max_length=200)
        answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        st.subheader(" Answer:")
        st.write(answer)


# chromadb==0.4.24 PyPDF2==3.0.1
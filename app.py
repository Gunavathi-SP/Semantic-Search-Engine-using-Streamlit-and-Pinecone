import streamlit as st
from sentence_transformers import SentenceTransformer
import tempfile
from pathlib import Path
from pinecone import Pinecone, ServerlessSpec

# -----------------------------
# CONFIG (ADD YOUR API KEY HERE)
# -----------------------------
import os

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "semantic-search-index"

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Semantic Search Engine",
    page_icon="🔍",
    layout="wide"
)

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# GENERATE EMBEDDINGS
# -----------------------------
def generate_embeddings(texts, model):
    return model.encode(texts)

# -----------------------------
# LOAD DOCUMENTS
# -----------------------------
def load_documents_from_file(filepath):
    with open(filepath, "r", encoding="utf-8") as file:
        lines = [line.strip() for line in file if line.strip()]

    documents = []
    for i in range(0, len(lines), 2):
        question = lines[i]
        answer = lines[i + 1] if i + 1 < len(lines) else ""
        documents.append(f"Q: {question}\nA: {answer}")

    return documents

# -----------------------------
# INITIALIZE PINECONE
# -----------------------------
@st.cache_resource
def init_pinecone():
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Create index if it doesn't exist
    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

    return pc.Index(INDEX_NAME)

# -----------------------------
# STORE DATA
# -----------------------------
def store_in_pinecone(index, documents, model):
    embeddings = generate_embeddings(documents, model)

    vectors = []
    for i, emb in enumerate(embeddings):
        vectors.append({
            "id": str(i),
            "values": emb.tolist(),
            "metadata": {"text": documents[i]}
        })

    index.upsert(vectors=vectors)

# -----------------------------
# RETRIEVE DATA
# -----------------------------
def retrieve(query, model, index, top_k=3):
    query_embedding = generate_embeddings([query], model)[0]

    results = index.query(
        vector=query_embedding.tolist(),
        top_k=top_k,
        include_metadata=True
    )

    output = []
    for match in results["matches"]:
        output.append({
            "document": match["metadata"]["text"],
            "score": match["score"]
        })

    return output

# -----------------------------
# MAIN APP
# -----------------------------
def main():

    st.title("🔍 Semantic Search Engine (Pinecone)")

    st.sidebar.title("⚙️ Settings")
    top_k = st.sidebar.slider("Number of Results", 1, 10, 3)

    uploaded_file = st.file_uploader("📄 Upload TXT file", type=["txt"])

    if uploaded_file is not None:

        # Save uploaded file
        temp_dir = tempfile.gettempdir()
        filepath = Path(temp_dir) / uploaded_file.name

        with open(filepath, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Load model
        with st.spinner("Loading model..."):
            model = load_model()

        # Load documents
        documents = load_documents_from_file(filepath)
        st.success(f"Loaded {len(documents)} documents")

        # Initialize Pinecone
        with st.spinner("Connecting to Pinecone..."):
            index = init_pinecone()

        # Store embeddings
        with st.spinner("Uploading data to Pinecone..."):
            store_in_pinecone(index, documents, model)

        # Query input
        query = st.text_input("💬 Enter your query")

        if query:
            with st.spinner("Searching..."):
                results = retrieve(query, model, index, top_k)

            st.subheader("Results")

            for res in results:
                st.write(f"Score: {res['score']:.4f}")
                st.write(res["document"])
                st.write("---")

    else:
        st.info("📌 Upload a file to start")

# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    main()
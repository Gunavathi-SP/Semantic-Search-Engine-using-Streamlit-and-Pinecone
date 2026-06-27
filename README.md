# 🔍 Semantic Search Engine using Streamlit and Pinecone

A Semantic Search Engine built using Streamlit and Pinecone that retrieves the most relevant documents based on the semantic meaning of a user's query instead of simple keyword matching.

## 🚀 Features

- 🔎 Semantic search using vector embeddings
- ⚡ Fast similarity search powered by Pinecone
- 🌐 Interactive Streamlit web interface
- 📄 Search across text documents
- 🎯 Retrieves contextually relevant results

---

## 🛠️ Tech Stack

- Python
- Streamlit
- Pinecone
- Sentence Transformers / Embedding Model
- NumPy

---

## 📂 Project Structure

```
Semantic-Search-Engine/
│
├── app.py              # Streamlit application
├── 1.txt               # Sample document
├── test.txt            # Sample document
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/your-username/semantic-search-engine.git
```

Move into the project directory

```bash
cd semantic-search-engine
```

Install dependencies

```bash
pip install -r requirements.txt
```

Configure your Pinecone API credentials.

Run the application

```bash
streamlit run app.py
```

---

## 📌 How It Works

1. Load text documents.
2. Generate embeddings for each document.
3. Store embeddings in Pinecone.
4. Convert the user's query into an embedding.
5. Perform similarity search.
6. Display the most relevant results.



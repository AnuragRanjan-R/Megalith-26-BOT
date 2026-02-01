import json
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# 1. Initialize the Vector DB with Local Embeddings
# This will download the model (approx 80MB) the first time you run it.
DB_PATH = "./chroma_db"
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vector_store = Chroma(
    collection_name="qna_store",
    embedding_function=embedding_function,
    persist_directory=DB_PATH
)

def ingest_data():
    # 2. Load Data
    try:
        with open("data/qna_dataset.json", "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("❌ Error: qna_dataset.json not found.")
        return

    # Ensure the top-level key 'faq' exists
    if "faq" not in data:
        print("❌ Error: JSON file does not contain 'faq' key.")
        return

    qna_data = data["faq"]

    documents = []
    print(f"🔄 Processing {len(qna_data)} Q&A pairs...")

    for entry in qna_data:
        if not isinstance(entry, dict):
            print("❌ Error: Each entry in 'faq' must be a dictionary.")
            continue

        question = entry.get("question")
        answer = entry.get("answer")

        if not question or not answer:
            print("⚠️ Skipping entry with missing 'question' or 'answer'.")
            continue

        # Embed Question, store Answer in metadata
        doc = Document(
            page_content=question, 
            metadata={"answer": answer}
        )
        documents.append(doc)

    # 3. Add to Vector DB
    if not documents:
        print("❌ No valid documents to process.")
        return

    print("💾 Saving to Vector Database...")
    vector_store.add_documents(documents)
    print("✅ Success! Database is ready.")

if __name__ == "__main__":
    ingest_data()
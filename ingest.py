import json
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

DB_PATH = "./chroma_db"

embedding_function = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

vector_store = Chroma(
    collection_name="megalith_knowledge",
    embedding_function=embedding_function,
    persist_directory=DB_PATH
)


def build_searchable_text(entry: dict) -> str:
    """Combine important fields into semantic text for embedding"""
    parts = []

    parts.append(f"Title: {entry.get('title', '')}")
    parts.append(f"Summary: {entry.get('summary', '')}")

    if entry.get("aliases"):
        parts.append("Aliases: " + ", ".join(entry["aliases"]))

    if entry.get("keywords"):
        parts.append("Keywords: " + ", ".join(entry["keywords"]))

    if entry.get("details"):
        parts.append("Details: " + entry["details"])

    if entry.get("rules"):
        parts.append("Rules: " + ", ".join(entry["rules"]))

    if entry.get("judging_criteria"):
        parts.append("Judging: " + ", ".join(entry["judging_criteria"]))

    return "\n".join(parts)


def ingest_data():
    try:
        with open("data/qna_dataset.json", "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("❌ qna_dataset.json not found.")
        return

    if not isinstance(data, list):
        print("❌ Dataset must be a LIST of knowledge objects.")
        return

    documents = []
    print(f"🔄 Processing {len(data)} knowledge entries...")

    for entry in data:
        if not isinstance(entry, dict):
            continue

        searchable_text = build_searchable_text(entry)

        metadata = {
            "id": entry.get("id"),
            "type": entry.get("type"),
            "title": entry.get("title"),
        }

        # Store FULL object as JSON string for retrieval
        metadata["full_data"] = json.dumps(entry, ensure_ascii=False)

        doc = Document(
            page_content=searchable_text,
            metadata=metadata
        )

        documents.append(doc)

    if not documents:
        print("❌ No valid documents.")
        return

    print("💾 Storing embeddings...")
    vector_store.add_documents(documents)
    print("✅ Knowledge base ready!")


if __name__ == "__main__":
    ingest_data()

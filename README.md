# Megalith Chatbot

## Overview
RAG-powered Q&A assistant for Megalith 2026.
- **Backend API** (FastAPI) — vector search + Groq LLM.
- **Frontend widget** — lives in `megalith-website/src/components/chatbot.tsx`.

## Quick Start

### 1. Setup
```bash
cd bot
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure
Create a `.env` file in the `bot/` folder:
```
GROQ_API_KEY=your_groq_api_key_here
```

Optional `.env` variables:
```
GROQ_MODEL=openai/gpt-oss-20b
GROQ_TEMPERATURE=0
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:3001,http://localhost:5173
```

### 3. Ingest Data
```bash
python ingest.py
```
This reads `data/qna_dataset.json` and builds the vector database in `chroma_db/`.

### 4. Run
```bash
python app.py
```
Or use the start script:
```bash
# Windows
start_bot.bat

# Linux/Mac
chmod +x start_bot.sh
./start_bot.sh
```

Server runs at **http://127.0.0.1:8000**

## Endpoints
| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Service info |
| GET | `/health` | Health check |
| POST | `/chat` | Send a question |

### POST /chat
**Request:**
```json
{ "question": "What is Megalith?" }
```
**Response:**
```json
{ "response": "...", "processing_time": 0.42 }
```

## Architecture
- **Embeddings:** `all-MiniLM-L6-v2` (HuggingFace)
- **Vector Store:** ChromaDB (local, at `./chroma_db`)
- **LLM:** Groq Chat API

## Error Responses
| Status | Meaning |
|--------|---------|
| 400 | Empty question |
| 500 | DB search or processing failure |
| 503 | Groq API unavailable (rate limit / outage) |

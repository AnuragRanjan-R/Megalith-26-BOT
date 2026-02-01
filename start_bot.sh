#!/bin/bash

echo "========================================"
echo "  Megalith Chatbot Backend Startup"
echo "========================================"
echo ""

# Check if virtual environment exists
if [ ! -f "venv/bin/activate" ]; then
    echo "[ERROR] Virtual environment not found!"
    echo "Please run: python3 -m venv venv"
    exit 1
fi

# Activate virtual environment
echo "[1/4] Activating virtual environment..."
source venv/bin/activate

# Check if dependencies are installed
python -c "import fastapi" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "[2/4] Installing dependencies..."
    pip install -r requirements.txt
else
    echo "[2/4] Dependencies already installed."
fi

# Check if .env exists in parent directory
if [ ! -f "../.env" ]; then
    echo ""
    echo "[WARNING] .env file not found in parent directory!"
    echo "Please create megalith2026-backend/.env with:"
    echo "  GROQ_API_KEY=your_groq_api_key_here"
    echo ""
    read -p "Press enter to continue anyway..."
fi

# Check if vector database exists
if [ ! -f "chroma_db/chroma.sqlite3" ]; then
    echo "[3/4] Vector database not found. Running ingest.py..."
    python ingest.py
    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to ingest data!"
        exit 1
    fi
else
    echo "[3/4] Vector database found."
fi

# Start the server
echo "[4/4] Starting FastAPI server..."
echo ""
echo "Server will be available at: http://127.0.0.1:8000"
echo "Press Ctrl+C to stop the server"
echo ""
uvicorn app:app --reload --host 127.0.0.1 --port 8000


@echo off
echo ========================================
echo   Megalith Chatbot Backend Startup
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found!
    echo Please run: python -m venv venv
    pause
    exit /b 1
)

REM Activate virtual environment
echo [1/4] Activating virtual environment...
call venv\Scripts\activate.bat

REM Check if dependencies are installed
python -c "import fastapi" 2>nul
if errorlevel 1 (
    echo [2/4] Installing dependencies...
    pip install -r requirements.txt
) else (
    echo [2/4] Dependencies already installed.
)

REM Check if .env exists in parent directory
if not exist "..\..env" (
    echo.
    echo [WARNING] .env file not found in parent directory!
    echo Please create megalith2026-backend\.env with:
    echo   GROQ_API_KEY=your_groq_api_key_here
    echo.
    pause
)

REM Check if vector database exists
if not exist "chroma_db\chroma.sqlite3" (
    echo [3/4] Vector database not found. Running ingest.py...
    python ingest.py
    if errorlevel 1 (
        echo [ERROR] Failed to ingest data!
        pause
        exit /b 1
    )
) else (
    echo [3/4] Vector database found.
)

REM Start the server
echo [4/4] Starting FastAPI server...
echo.
echo Server will be available at: http://127.0.0.1:8000
echo Press Ctrl+C to stop the server
echo.
uvicorn app:app --reload --host 127.0.0.1 --port 8000

pause


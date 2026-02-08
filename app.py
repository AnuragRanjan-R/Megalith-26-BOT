import os
import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from pathlib import Path
from typing import Optional
import time


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load .env from current directory (bot/.env)
# Since bot is now an independent root, look for .env in its own directory
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

# Fallback: also check parent directory for backward compatibility
if not env_path.exists():
    parent_env = Path(__file__).parent.parent / ".env"
    if parent_env.exists():
        load_dotenv(dotenv_path=parent_env)
    else:
        load_dotenv()  # Fallback to default .env location

app = FastAPI(
    title="Megalith Chatbot API",
    description="RAG-powered Q&A chatbot for Megalith 2026",
    version="1.0.0"
)

# CORS configuration - allow multiple origins
allowed_origins = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://localhost:3001,http://localhost:5173"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)
# 1. Setup Embeddings (Must match ingest.py exactly)
DB_PATH = "./chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

try:
    embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = Chroma(
        collection_name="megalith_knowledge", 
        embedding_function=embedding_function, 
        persist_directory=DB_PATH
    )
    logger.info("Vector store initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize vector store: {e}")
    raise

# 2. Setup Groq LLM
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    logger.error("GROQ_API_KEY not found in environment variables")
    raise ValueError(
        "GROQ_API_KEY not found in environment variables. "
        "Please create a .env file in bot/.env with GROQ_API_KEY=your_key"
    )

try:
    llm = ChatGroq(
        model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
        temperature=float(os.getenv("GROQ_TEMPERATURE", "0")),
        groq_api_key=groq_api_key,
        timeout=30.0
    )
    logger.info("Groq LLM initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Groq LLM: {e}")
    raise

# Prompt template (cached)
PROMPT_TEMPLATE = ChatPromptTemplate.from_template("""
You are the official AI assistant for Megalith 2026.

Only answer Megalith-related queries using the context below.
If the answer is not in the context, respond exactly:
"I don't have that information. Please contact the organizers for more details."

If the question is unrelated to Megalith, respond exactly:
"I'm here to help only with Megalith 2026 queries. Ask me about events, registration, accommodation, schedules, teams, or sponsors."

Response style:
- Be brief (2 to 4 sentences).
- If listing items, show a short list (max 6 items).
- End with: "Ask me for details on any item."
- Use Markdown formatting. Use a table only if the user asks for a table.

Context:
{context}

User Question: {question}

Answer:
""")

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500, description="User's question")

class ChatResponse(BaseModel):
    response: str
    processing_time: Optional[float] = None

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "chatbot"}

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Megalith Chatbot API",
        "version": "1.0.0",
        "status": "running"
    }

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: QueryRequest):
    """
    Chat endpoint for Q&A queries
    """
    start_time = time.time()
    
    try:
        # Validate question
        question = request.question.strip()
        if not question:
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        logger.info(f"Processing question: {question[:50]}...")
        
        # Search for relevant questions (with error handling)
        try:
            results = vector_store.similarity_search(question, k=3)
        except Exception as e:
            logger.error(f"Vector search error: {e}")
            raise HTTPException(
                status_code=500,
                detail="Database search failed. Please try again later."
            )
        
        if not results:
            return ChatResponse(
                response="I couldn't find relevant information in the database. Please try rephrasing your question or contact the organizers.",
                processing_time=time.time() - start_time
            )

        # Prepare context
        context_text = "\n\n".join(
            doc.page_content for doc in results
        )

        # Generate prompt
        try:
            prompt = PROMPT_TEMPLATE.invoke({
                "context": context_text,
                "question": question
            })
        except Exception as e:
            logger.error(f"Prompt generation error: {e}")
            raise HTTPException(
                status_code=500,
                detail="Failed to process your question. Please try again."
            )
        
        # Generate response with Groq
        try:
            response = llm.invoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            logger.error(f"LLM invocation error: {e}")
            raise HTTPException(
                status_code=503,
                detail="AI service is temporarily unavailable. Please try again later."
            )
        
        processing_time = time.time() - start_time
        logger.info(f"Response generated in {processing_time:.2f}s")
        
        return ChatResponse(
            response=response_text,
            processing_time=round(processing_time, 2)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred. Please try again later."
        )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )
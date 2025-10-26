from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import google.generativeai as genai
from datetime import datetime
import os
from dotenv import load_dotenv
import PyPDF2
import io
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import wikipedia
import json

# Load environment variables
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-2.5-flash')
embedding_model = "models/text-embedding-004"

# Initialize FastAPI
app = FastAPI(title="RAG Chat API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage
document_chunks = []
chunk_embeddings = []
chat_history = []

# Models
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    confidence: float

class HistoryItem(BaseModel):
    question: str
    answer: str
    timestamp: str

class UploadResponse(BaseModel):
    message: str
    chunks: int

# Helper functions
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    
    return chunks

def get_embedding(text: str) -> List[float]:
    """Get embedding for text using Gemini."""
    try:
        result = genai.embed_content(
            model=embedding_model,
            content=text,
            task_type="retrieval_document"
        )
        return result['embedding']
    except Exception as e:
        print(f"Embedding error: {e}")
        # text-embedding-004 returns 768-dimensional vectors
        return [0.0] * 768

def extract_text_from_pdf(file_content: bytes) -> str:
    """Extract text from PDF file."""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read PDF: {str(e)}")

def retrieve_relevant_chunks(query: str, top_k: int = 3) -> tuple[List[str], float]:
    """Retrieve most relevant chunks for a query."""
    if not document_chunks:
        return [], 0.0
    
    # Get query embedding
    query_embedding = get_embedding(query)
    query_array = np.array(query_embedding).reshape(1, -1)
    
    # Calculate similarities
    chunk_array = np.array(chunk_embeddings)
    similarities = cosine_similarity(query_array, chunk_array)[0]
    
    # Get top-k chunks
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    relevant_chunks = [document_chunks[i] for i in top_indices]
    max_similarity = float(similarities[top_indices[0]])
    
    return relevant_chunks, max_similarity

def generate_answer(question: str, context_chunks: List[str]) -> str:
    """Generate answer using Gemini with retrieved context."""
    context = "\n\n".join([f"[Context {i+1}]: {chunk}" for i, chunk in enumerate(context_chunks)])
    
    prompt = f"""You are a helpful assistant that answers questions based on the provided context.

Context:
{context}

Question: {question}

Instructions:
- Answer the question based ONLY on the information in the context above
- If the context doesn't contain enough information to answer the question, say so
- Be concise and specific
- Cite which context sections you used if relevant

Answer:"""

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate answer: {str(e)}")

# API Endpoints
@app.get("/")
async def root():
    return {"message": "RAG Chat API", "status": "running"}

@app.post("/api/documents/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document."""
    global document_chunks, chunk_embeddings
    
    try:
        # Read file content
        content = await file.read()
        
        # Extract text based on file type
        if file.filename.endswith('.pdf'):
            text = extract_text_from_pdf(content)
        elif file.filename.endswith('.txt'):
            text = content.decode('utf-8')
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Use PDF or TXT.")
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="No text could be extracted from the file.")
        
        # Chunk the text
        document_chunks = chunk_text(text)
        
        # Generate embeddings for all chunks
        chunk_embeddings = [get_embedding(chunk) for chunk in document_chunks]
        
        return UploadResponse(
            message=f"Successfully processed {file.filename}",
            chunks=len(document_chunks)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.post("/api/documents/wikipedia", response_model=UploadResponse)
async def load_wikipedia(topic: str):
    """Load content from Wikipedia."""
    global document_chunks, chunk_embeddings
    
    try:
        # Search and get Wikipedia page
        page = wikipedia.page(topic, auto_suggest=True)
        text = page.content
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="No content found for this topic.")
        
        # Chunk the text
        document_chunks = chunk_text(text)
        
        # Generate embeddings
        chunk_embeddings = [get_embedding(chunk) for chunk in document_chunks]
        
        return UploadResponse(
            message=f"Successfully loaded Wikipedia: {topic}",
            chunks=len(document_chunks)
        )
        
    except wikipedia.exceptions.DisambiguationError as e:
        raise HTTPException(
            status_code=400, 
            detail=f"Topic is ambiguous. Please be more specific. Options: {', '.join(e.options[:5])}"
        )
    except wikipedia.exceptions.PageError:
        raise HTTPException(status_code=404, detail="Wikipedia page not found for this topic.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading Wikipedia: {str(e)}")

@app.post("/api/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query the loaded documents."""
    if not document_chunks:
        raise HTTPException(
            status_code=400, 
            detail="No document loaded. Please upload a document or load a Wikipedia page first."
        )
    
    try:
        # Retrieve relevant chunks
        relevant_chunks, confidence = retrieve_relevant_chunks(request.question, top_k=3)
        
        # Generate answer
        answer = generate_answer(request.question, relevant_chunks)
        
        # Save to history
        chat_history.append({
            "question": request.question,
            "answer": answer,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only last 50 items
        if len(chat_history) > 50:
            chat_history.pop(0)
        
        return QueryResponse(
            answer=answer,
            sources=relevant_chunks,
            confidence=confidence
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/api/history", response_model=List[HistoryItem])
async def get_history():
    """Get chat history."""
    return chat_history

@app.delete("/api/history")
async def clear_history():
    """Clear chat history."""
    global chat_history
    chat_history = []
    return {"message": "History cleared"}

@app.delete("/api/documents")
async def clear_documents():
    """Clear loaded documents."""
    global document_chunks, chunk_embeddings
    document_chunks = []
    chunk_embeddings = []
    return {"message": "Documents cleared"}

@app.get("/api/status")
async def get_status():
    """Get current system status."""
    return {
        "documents_loaded": len(document_chunks) > 0,
        "num_chunks": len(document_chunks),
        "history_count": len(chat_history)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import os
from pathlib import Path
import shutil
import logging

from pdf_processor import PDFProcessor
from vector_store import VectorStoreManager
from query_engine import QueryEngine
from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG PDF Intelligence System",
    description="AI-powered document intelligence with semantic search",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pdf_processor = PDFProcessor()
vector_store_manager = VectorStoreManager()
query_engine = QueryEngine()

UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

class QueryRequest(BaseModel):
    query: str
    document_ids: Optional[List[str]] = None
    top_k: int = 5

class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]
    confidence: float

class SummarizeRequest(BaseModel):
    document_id: str
    summary_type: str = "brief"

class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    page_count: int
    chunk_count: int
    upload_time: str

@app.get("/")
async def root():
    return {
        "status": "active",
        "service": "RAG PDF Intelligence System",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "components": {
            "pdf_processor": "active",
            "vector_store": "active",
            "query_engine": "active",
            "gemini_api": "connected"
        }
    }

@app.post("/upload", response_model=List[DocumentInfo])
async def upload_documents(files: List[UploadFile] = File(...)):
    try:
        uploaded_docs = []
        
        for file in files:
            if not file.filename.endswith('.pdf'):
                raise HTTPException(
                    status_code=400,
                    detail=f"File {file.filename} is not a PDF"
                )
            
            file_path = UPLOAD_DIR / file.filename
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            logger.info(f"Processing PDF: {file.filename}")
            
            doc_info = pdf_processor.process_pdf(str(file_path), file.filename)
            
            vector_store_manager.add_document(
                doc_info['document_id'],
                doc_info['chunks'],
                doc_info['metadata']
            )
            
            uploaded_docs.append(DocumentInfo(**doc_info))
            
            logger.info(f"Successfully processed: {file.filename}")
        
        return uploaded_docs
    
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    try:
        logger.info(f"Processing query: {request.query}")
        
        relevant_chunks = vector_store_manager.search(
            query=request.query,
            top_k=request.top_k,
            document_ids=request.document_ids
        )
        
        if not relevant_chunks:
            return QueryResponse(
                answer="No relevant information found in the documents.",
                sources=[],
                confidence=0.0
            )
        
        response = query_engine.generate_answer(
            query=request.query,
            context_chunks=relevant_chunks
        )
        
        return QueryResponse(**response)
    
    except Exception as e:
        logger.error(f"Query error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summarize")
async def summarize_document(request: SummarizeRequest):
    try:
        logger.info(f"Summarizing document: {request.document_id}")
        
        chunks = vector_store_manager.get_document_chunks(request.document_id)
        
        if not chunks:
            raise HTTPException(
                status_code=404,
                detail=f"Document {request.document_id} not found"
            )
        
        summary = query_engine.summarize_text(
            chunks=chunks,
            summary_type=request.summary_type
        )
        
        return {
            "document_id": request.document_id,
            "summary": summary,
            "summary_type": request.summary_type
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Summarization error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents", response_model=List[DocumentInfo])
async def list_documents():
    try:
        documents = vector_store_manager.list_documents()
        return [DocumentInfo(**doc) for doc in documents]
    except Exception as e:
        logger.error(f"List documents error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    try:
        success = vector_store_manager.delete_document(document_id)
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Document {document_id} not found"
            )
        
        return {
            "message": f"Document {document_id} deleted successfully",
            "document_id": document_id
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clear")
async def clear_all_documents():
    try:
        vector_store_manager.clear_all()
        
        for file in UPLOAD_DIR.glob("*.pdf"):
            file.unlink()
        
        return {"message": "All documents cleared successfully"}
    except Exception as e:
        logger.error(f"Clear error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
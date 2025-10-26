# RAG Chat Application

A Retrieval-Augmented Generation chat system powered by Google Gemini AI. Upload documents or load Wikipedia articles to ask questions and receive context-aware answers.

![Upload Interface](ss/upload.png)
![Chat Interface](ss/chat.png)

## Features

- Multi-file PDF upload and processing
- Semantic search with FAISS vector database
- Context-aware Q&A using RAG architecture
- Document summarization (brief, detailed, bullet points)
- Fast retrieval with sentence embeddings
- Scalable FastAPI backend

## Technology

- FastAPI - High-performance REST API framework
- PyTorch - Deep learning framework for embeddings
- FAISS - Facebook AI Similarity Search for vector database
- LangChain - Text processing and chunking pipeline
- Sentence Transformers - Semantic embeddings (all-MiniLM-L6-v2)
- Gemini API - Google's LLM for answer generation

## Setup Instructions

### 1. Create Virtual Environment

Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API Key

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_api_key_here
```

Get your API key from https://aistudio.google.com/api-keys

### 4. Start Backend Server

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

The API will run on http://localhost:8000

### 5. Start Frontend Server

Using VS Code Live Server:
- Install Live Server extension
- Right-click `index.html`
- Select "Open with Live Server"

Or using Python:
```bash
python -m http.server 5500
```

Access the application at http://localhost:5500

## Usage

1. Upload a document or enter a Wikipedia topic
2. Wait for processing to complete
3. Type questions in the chat interface
4. View answers with confidence scores and source references

## API Endpoints

```
POST /api/documents/upload          Upload document
POST /api/documents/wikipedia       Load Wikipedia article
POST /api/query                     Submit question
GET  /api/history                   Retrieve chat history
GET  /api/status                    System status
```
## Requirements

- Python 3.8 or higher
- Google Gemini API key

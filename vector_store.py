import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import json
import os
from pathlib import Path
import logging

from config import settings

logger = logging.getLogger(__name__)

class VectorStoreManager:
    def __init__(self):
        logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
        self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
        
        self.dimension = settings.VECTOR_DIMENSION
        self.index = faiss.IndexFlatL2(self.dimension)
        
        self.metadata_store = {}
        self.document_chunks = {}
        
        Path(settings.FAISS_INDEX_PATH).parent.mkdir(exist_ok=True)
        
        self._load_index()
    
    def _load_index(self):
        try:
            index_path = f"{settings.FAISS_INDEX_PATH}/index.faiss"
            metadata_path = settings.METADATA_PATH
            
            if os.path.exists(index_path) and os.path.exists(metadata_path):
                self.index = faiss.read_index(index_path)
                
                with open(metadata_path, 'r') as f:
                    data = json.load(f)
                    self.metadata_store = data.get('metadata_store', {})
                    self.document_chunks = data.get('document_chunks', {})
                
                logger.info(f"Loaded existing index with {self.index.ntotal} vectors")
        except Exception as e:
            logger.warning(f"Could not load existing index: {str(e)}")
    
    def _save_index(self):
        try:
            os.makedirs(settings.FAISS_INDEX_PATH, exist_ok=True)
            
            index_path = f"{settings.FAISS_INDEX_PATH}/index.faiss"
            faiss.write_index(self.index, index_path)
            
            with open(settings.METADATA_PATH, 'w') as f:
                json.dump({
                    'metadata_store': self.metadata_store,
                    'document_chunks': self.document_chunks
                }, f, indent=2)
            
            logger.info("Index and metadata saved successfully")
        except Exception as e:
            logger.error(f"Error saving index: {str(e)}")
            raise
    
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        try:
            embeddings = self.embedding_model.encode(
                texts,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            return embeddings.astype('float32')
        except Exception as e:
            logger.error(f"Error creating embeddings: {str(e)}")
            raise
    
    def add_document(self, document_id: str, chunks: List[Dict], metadata: Dict):
        try:
            texts = [chunk['text'] for chunk in chunks]
            
            logger.info(f"Creating embeddings for {len(texts)} chunks")
            embeddings = self.create_embeddings(texts)
            
            start_idx = self.index.ntotal
            self.index.add(embeddings)
            
            for idx, chunk in enumerate(chunks):
                vector_id = start_idx + idx
                self.metadata_store[str(vector_id)] = {
                    'document_id': document_id,
                    'chunk_id': chunk['chunk_id'],
                    'text': chunk['text'],
                    'metadata': chunk['metadata']
                }
            
            self.document_chunks[document_id] = {
                'metadata': metadata,
                'chunk_count': len(chunks),
                'vector_ids': list(range(start_idx, start_idx + len(chunks)))
            }
            
            self._save_index()
            
            logger.info(f"Added document {document_id} with {len(chunks)} chunks")
        
        except Exception as e:
            logger.error(f"Error adding document: {str(e)}")
            raise
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        document_ids: Optional[List[str]] = None
    ) -> List[Dict]:
        try:
            if self.index.ntotal == 0:
                logger.warning("Index is empty")
                return []
            
            query_embedding = self.create_embeddings([query])
            
            distances, indices = self.index.search(query_embedding, min(top_k * 2, self.index.ntotal))
            
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx == -1:
                    continue
                
                metadata = self.metadata_store.get(str(idx))
                if not metadata:
                    continue
                
                if document_ids and metadata['document_id'] not in document_ids:
                    continue
                
                results.append({
                    'text': metadata['text'],
                    'document_id': metadata['document_id'],
                    'chunk_id': metadata['chunk_id'],
                    'metadata': metadata['metadata'],
                    'score': float(dist),
                    'similarity': 1 / (1 + float(dist))
                })
                
                if len(results) >= top_k:
                    break
            
            logger.info(f"Found {len(results)} relevant chunks")
            return results
        
        except Exception as e:
            logger.error(f"Error during search: {str(e)}")
            raise
    
    def get_document_chunks(self, document_id: str) -> List[str]:
        try:
            doc_info = self.document_chunks.get(document_id)
            if not doc_info:
                return []
            
            chunks = []
            for vector_id in doc_info['vector_ids']:
                metadata = self.metadata_store.get(str(vector_id))
                if metadata:
                    chunks.append(metadata['text'])
            
            return chunks
        except Exception as e:
            logger.error(f"Error retrieving document chunks: {str(e)}")
            raise
    
    def list_documents(self) -> List[Dict]:
        documents = []
        for doc_id, doc_info in self.document_chunks.items():
            documents.append({
                'document_id': doc_id,
                'filename': doc_info['metadata'].get('filename', 'unknown'),
                'page_count': doc_info['metadata'].get('page_count', 0),
                'chunk_count': doc_info['chunk_count'],
                'upload_time': doc_info['metadata'].get('upload_time', '')
            })
        return documents
    
    def delete_document(self, document_id: str) -> bool:
        try:
            if document_id not in self.document_chunks:
                return False
            
            del self.document_chunks[document_id]
            
            keys_to_remove = [
                k for k, v in self.metadata_store.items()
                if v['document_id'] == document_id
            ]
            for key in keys_to_remove:
                del self.metadata_store[key]
            
            self._save_index()
            logger.info(f"Deleted document {document_id}")
            return True
        
        except Exception as e:
            logger.error(f"Error deleting document: {str(e)}")
            raise
    
    def clear_all(self):
        self.index = faiss.IndexFlatL2(self.dimension)
        self.metadata_store = {}
        self.document_chunks = {}
        self._save_index()
        logger.info("Cleared all documents")
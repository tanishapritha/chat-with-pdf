import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict
import hashlib
from datetime import datetime
import logging

from config import settings

logger = logging.getLogger(__name__)

class PDFProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def extract_text_from_pdf(self, pdf_path: str) -> tuple[str, int]:
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                page_count = len(pdf_reader.pages)
                
                text = ""
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    text += f"\n--- Page {page_num + 1} ---\n{page_text}"
                
                logger.info(f"Extracted {len(text)} characters from {page_count} pages")
                return text, page_count
        
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise
    
    def create_chunks(self, text: str, metadata: Dict) -> List[Dict]:
        try:
            text_chunks = self.text_splitter.split_text(text)
            
            chunks = []
            for idx, chunk in enumerate(text_chunks):
                chunks.append({
                    'text': chunk,
                    'chunk_id': idx,
                    'metadata': {
                        **metadata,
                        'chunk_index': idx,
                        'total_chunks': len(text_chunks)
                    }
                })
            
            logger.info(f"Created {len(chunks)} chunks from document")
            return chunks
        
        except Exception as e:
            logger.error(f"Error creating chunks: {str(e)}")
            raise
    
    def generate_document_id(self, filename: str, content: str) -> str:
        unique_string = f"{filename}_{content[:1000]}_{datetime.now().isoformat()}"
        return hashlib.md5(unique_string.encode()).hexdigest()
    
    def process_pdf(self, pdf_path: str, filename: str) -> Dict:
        try:
            text, page_count = self.extract_text_from_pdf(pdf_path)
            
            if not text.strip():
                raise ValueError("No text could be extracted from PDF")
            
            document_id = self.generate_document_id(filename, text)
            
            metadata = {
                'document_id': document_id,
                'filename': filename,
                'page_count': page_count,
                'upload_time': datetime.now().isoformat(),
                'text_length': len(text)
            }
            
            chunks = self.create_chunks(text, metadata)
            
            return {
                'document_id': document_id,
                'filename': filename,
                'page_count': page_count,
                'chunk_count': len(chunks),
                'upload_time': metadata['upload_time'],
                'chunks': chunks,
                'metadata': metadata
            }
        
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            raise
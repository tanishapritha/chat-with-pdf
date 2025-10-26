import google.generativeai as genai
from typing import List, Dict
import logging

from config import settings

logger = logging.getLogger(__name__)

class QueryEngine:
    def __init__(self):
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.model = genai.GenerativeModel(settings.GEMINI_MODEL)
        
        self.generation_config = {
            'temperature': settings.TEMPERATURE,
            'top_p': 0.95,
            'top_k': 40,
            'max_output_tokens': 2048,
        }
        
        logger.info(f"Initialized Query Engine with {settings.GEMINI_MODEL}")
    
    def _prepare_context(self, chunks: List[Dict], max_length: int = None) -> str:
        if not chunks:
            return ""
        
        max_length = max_length or settings.MAX_CONTEXT_LENGTH
        
        context_parts = []
        current_length = 0
        
        for idx, chunk in enumerate(chunks, 1):
            chunk_text = f"[Source {idx} - {chunk['metadata']['filename']}]:\n{chunk['text']}\n"
            chunk_length = len(chunk_text)
            
            if current_length + chunk_length > max_length:
                break
            
            context_parts.append(chunk_text)
            current_length += chunk_length
        
        return "\n".join(context_parts)
    
    def generate_answer(self, query: str, context_chunks: List[Dict]) -> Dict:
        try:
            context = self._prepare_context(context_chunks)
            
            if not context:
                return {
                    'answer': "I couldn't find relevant information to answer your question.",
                    'sources': [],
                    'confidence': 0.0
                }
            
            prompt = f"""You are an intelligent document assistant. Answer the user's question based ONLY on the provided context from the documents.

Context from documents:
{context}

User Question: {query}

Instructions:
1. Answer the question accurately using information from the context
2. If the context doesn't contain enough information, say so clearly
3. Be concise and specific
4. Cite which source(s) you're using in your answer
5. If asked for specific details, provide them exactly as they appear in the context

Answer:"""

            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            
            answer = response.text
            
            avg_similarity = sum(chunk['similarity'] for chunk in context_chunks) / len(context_chunks)
            confidence = min(avg_similarity, 1.0)
            
            sources = []
            seen_docs = set()
            for chunk in context_chunks:
                doc_id = chunk['document_id']
                if doc_id not in seen_docs:
                    sources.append({
                        'document_id': doc_id,
                        'filename': chunk['metadata']['filename'],
                        'chunk_id': chunk['chunk_id'],
                        'relevance_score': round(chunk['similarity'], 3)
                    })
                    seen_docs.add(doc_id)
            
            logger.info(f"Generated answer with confidence: {confidence:.3f}")
            
            return {
                'answer': answer,
                'sources': sources,
                'confidence': round(confidence, 3)
            }
        
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            raise
    
    def summarize_text(self, chunks: List[str], summary_type: str = "brief") -> str:
        try:
            full_text = "\n\n".join(chunks)
            
            if len(full_text) > settings.MAX_CONTEXT_LENGTH:
                full_text = full_text[:settings.MAX_CONTEXT_LENGTH] + "..."
            
            if summary_type == "brief":
                prompt = f"""Provide a brief summary (2-3 sentences) of the following document:

{full_text}

Brief Summary:"""
            
            elif summary_type == "detailed":
                prompt = f"""Provide a comprehensive summary of the following document. Include:
- Main topics and themes
- Key findings or arguments
- Important details and conclusions

Document:
{full_text}

Detailed Summary:"""
            
            elif summary_type == "bullet_points":
                prompt = f"""Summarize the following document as bullet points covering:
- Main topics
- Key facts and figures
- Important conclusions

Document:
{full_text}

Bullet Point Summary:"""
            
            else:
                prompt = f"""Summarize the following document:

{full_text}

Summary:"""
            
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            
            summary = response.text
            logger.info(f"Generated {summary_type} summary")
            
            return summary
        
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            raise
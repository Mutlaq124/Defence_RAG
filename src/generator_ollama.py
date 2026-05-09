'''
Ensure generator's final input/output token counts are within limits
LATER -> KV cache store (previous context memory), paged attention
LATER -> token usage tracking (input/output tokens) for cost monitoring
'''

import requests
import logging
from . import config
import json

logger = logging.getLogger(__name__)

class PromptTemplate:
    RAG_TEMPLATE = """Act as Expert Technical Manual Fighter jet assistant, and follow these instructions:
    CRITICAL INSTRUCTIONS:
1. Answer ONLY using information from the provided context
2. If the context doesn't contain the answer, respond: "I cannot answer this question based on the provided context."
3. Be precise and cite specific details from the context
4. Do not make any assumptions or use external or internal knowledge

CONTEXT: {context}
QUESTION: {question}
ANSWER (be concise):"""

# Chunk ID -> 

class OllamaGenerator:
    def __init__(self, model=None):
        self.model = model or config.GEN_MODEL
        self.url = f"{config.OLLAMA_BASE_URL}/api/generate"
    
    def generate(self, prompt: str, max_tokens: int = None, temperature: float = 0.7):
        max_tokens = max_tokens or config.MAX_RESPONSE_TOKENS
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False, # No streaming for simplicity
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature
            }
        }
        
        try:
            resp = requests.post(self.url, json=payload)
            resp.raise_for_status()
            result = resp.json()
            
            if "response" not in result:
                logger.error("invalid_response_format", result=result)
                raise ValueError("Missing 'response' field")
            
            return result["response"].strip()
            
        except Exception as e:
            logger.error("generation_failed", error=str(e))
            raise

    def generate_with_template(
        self,
        question: str,
        context: str,
        max_tokens: int = None,
        temperature: float = 0.7
    ):
        prompt = PromptTemplate.RAG_TEMPLATE.format(
            context=context,
            question=question
        )
        
        return self.generate(prompt, max_tokens, temperature)
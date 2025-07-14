import os
import json
import aiohttp
import asyncio
from typing import Any, List, Optional, Dict
from llama_index.core.embeddings import BaseEmbedding
from pydantic import Field


class OllamaEmbedding(BaseEmbedding):
    """Ollama embedding wrapper for LlamaIndex."""
    model: str = Field(default="nomic-embed-text")
    base_url: str = Field(default="http://localhost:11434")
    
    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        **kwargs: Any,
    ):
        super().__init__(model=model, base_url=base_url, **kwargs)
        
    @property
    def metadata(self) -> Dict[str, Any]:
        """Get embedding metadata."""
        return {
            "model_name": self.model,
            "embed_dim": 768,  # nomic-embed-text dimension
        }
    
    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        return asyncio.run(self._aget_query_embedding(query))
    
    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        return asyncio.run(self._aget_text_embedding(text))
    
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get text embeddings."""
        return asyncio.run(self._aget_text_embeddings(texts))
    
    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Async get query embedding."""
        return await self._aget_text_embedding(query)
    
    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Async get text embedding."""
        async with aiohttp.ClientSession() as session:
            payload = {
                "model": self.model,
                "prompt": text,
            }
            
            async with session.post(
                f"{self.base_url}/api/embeddings",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status != 200:
                    raise Exception(f"Ollama embeddings API error: {response.status}")
                
                result = await response.json()
                return result["embedding"]
    
    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Async get text embeddings."""
        embeddings = []
        for text in texts:
            embedding = await self._aget_text_embedding(text)
            embeddings.append(embedding)
        return embeddings 
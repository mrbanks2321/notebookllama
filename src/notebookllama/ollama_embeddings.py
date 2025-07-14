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
        print(f"[_get_text_embeddings] Start: num_texts={len(texts)}")
        import asyncio
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import threading
            result = []
            def run():
                print("[_get_text_embeddings] In thread, before asyncio.run")
                result.append(asyncio.run(self._aget_text_embeddings(texts)))
                print("[_get_text_embeddings] In thread, after asyncio.run")
            t = threading.Thread(target=run)
            t.start()
            t.join()
            print(f"[_get_text_embeddings] Returning from thread: {type(result[0])}")
            return result[0]
        else:
            print("[_get_text_embeddings] Before asyncio.run")
            out = asyncio.run(self._aget_text_embeddings(texts))
            print(f"[_get_text_embeddings] After asyncio.run, type={type(out)}")
            return out
    
    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Async get query embedding."""
        return await self._aget_text_embedding(query)
    
    async def _aget_text_embedding(self, text: str) -> List[float]:
        print(f"[_aget_text_embedding] Start: text length={len(text)}")
        import aiohttp
        async with aiohttp.ClientSession() as session:
            payload = {
                "model": self.model,
                "prompt": text,
            }
            print(f"[_aget_text_embedding] Sending POST to {self.base_url}/api/embeddings with model={self.model}")
            async with session.post(
                f"{self.base_url}/api/embeddings",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                print(f"[_aget_text_embedding] Response status: {response.status}")
                if response.status != 200:
                    raise Exception(f"Ollama embeddings API error: {response.status}")
                result = await response.json()
                print(f"[_aget_text_embedding] Received embedding of length {len(result['embedding'])}")
                return result["embedding"]
    
    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        print(f"[_aget_text_embeddings] Start: num_texts={len(texts)}")
        embeddings = []
        for text in texts:
            print(f"[_aget_text_embeddings] Getting embedding for text of length {len(text)}")
            embedding = await self._aget_text_embedding(text)
            print(f"[_aget_text_embeddings] Got embedding of length {len(embedding)}")
            embeddings.append(embedding)
        print(f"[_aget_text_embeddings] Returning {len(embeddings)} embeddings")
        return embeddings 
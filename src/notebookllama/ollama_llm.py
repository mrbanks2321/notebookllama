import os
import json
import aiohttp
import asyncio
from typing import Any, List, Optional, Dict
from llama_index.core.llms import ChatMessage, CompletionResponse, CompletionResponseGen, LLM, ChatResponse, ChatResponseGen
from pydantic import Field


class OllamaLLM(LLM):
    """Ollama LLM wrapper for LlamaIndex."""
    model: str = Field(default="gemma3:4b")
    base_url: str = Field(default="http://localhost:11434")
    temperature: float = Field(default=0.1)
    max_tokens: Optional[int] = Field(default=None)
    
    def __init__(
        self,
        model: str = "gemma3:4b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ):
        super().__init__(model=model, base_url=base_url, temperature=temperature, max_tokens=max_tokens, **kwargs)
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Get LLM metadata."""
        return {
            "model_name": self.model,
            "is_chat_model": True,
            "is_function_calling_model": False,
        }
    
    def _format_messages_to_prompt(self, messages: List[ChatMessage]) -> str:
        """Convert chat messages to a single prompt string."""
        prompt = ""
        for message in messages:
            if message.role == "system":
                prompt += f"System: {message.content}\n\n"
            elif message.role == "user":
                prompt += f"User: {message.content}\n\n"
            elif message.role == "assistant":
                prompt += f"Assistant: {message.content}\n\n"
        prompt += "Assistant: "
        return prompt
    
    def _chat_to_completion(self, messages: List[ChatMessage]) -> str:
        """Convert chat messages to completion format."""
        return self._format_messages_to_prompt(messages)
    
    async def acomplete(
        self, prompt: str, **kwargs: Any
    ) -> CompletionResponse:
        """Async completion."""
        async with aiohttp.ClientSession() as session:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": kwargs.get("temperature", self.temperature),
                }
            }
            
            if self.max_tokens:
                payload["options"]["num_predict"] = self.max_tokens
                
            async with session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status != 200:
                    raise Exception(f"Ollama API error: {response.status}")
                
                result = await response.json()
                return CompletionResponse(text=result["response"])
    
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """Sync completion."""
        return asyncio.run(self.acomplete(prompt, **kwargs))
    
    async def achat(self, messages: List[ChatMessage], **kwargs: Any) -> ChatResponse:
        """Async chat completion."""
        prompt = self._chat_to_completion(messages)
        completion = await self.acomplete(prompt, **kwargs)
        return ChatResponse(message=ChatMessage(role="assistant", content=completion.text))
    
    def chat(self, messages: List[ChatMessage], **kwargs: Any) -> ChatResponse:
        """Sync chat completion."""
        return asyncio.run(self.achat(messages, **kwargs))
    
    async def astream_complete(
        self, prompt: str, **kwargs: Any
    ) -> CompletionResponseGen:
        """Async streaming completion."""
        async with aiohttp.ClientSession() as session:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": kwargs.get("temperature", self.temperature),
                }
            }
            
            if self.max_tokens:
                payload["options"]["num_predict"] = self.max_tokens
                
            async with session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status != 200:
                    raise Exception(f"Ollama API error: {response.status}")
                
                async for line in response.content:
                    if line:
                        try:
                            data = json.loads(line.decode().strip())
                            if "response" in data:
                                yield CompletionResponse(text=data["response"], delta=data["response"])
                        except json.JSONDecodeError:
                            continue
    
    async def astream_chat(
        self, messages: List[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        """Async streaming chat completion."""
        prompt = self._chat_to_completion(messages)
        async for completion in self.astream_complete(prompt, **kwargs):
            yield ChatResponse(
                message=ChatMessage(role="assistant", content=completion.text),
                delta=completion.delta
            ) 

    def stream_complete(self, prompt: str, **kwargs: Any):
        """Sync streaming completion (not implemented)."""
        raise NotImplementedError("stream_complete is not implemented for OllamaLLM.")

    def stream_chat(self, messages: List[ChatMessage], **kwargs: Any):
        """Sync streaming chat completion (not implemented)."""
        raise NotImplementedError("stream_chat is not implemented for OllamaLLM.") 
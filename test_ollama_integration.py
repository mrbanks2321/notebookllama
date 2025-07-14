#!/usr/bin/env python3
"""
Test script for Ollama integration components.
"""

import asyncio
import sys
import os
import traceback

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from notebookllama.ollama_llm import OllamaLLM
from notebookllama.ollama_embeddings import OllamaEmbedding
from notebookllama.local_parsers import LocalParser
from notebookllama.local_extraction import LocalExtractor, SourceText


async def test_ollama_llm():
    """Test Ollama LLM wrapper."""
    print("Testing Ollama LLM...")
    
    try:
        llm = OllamaLLM(model="gemma3:4b", temperature=0.1)
        
        # Test basic completion
        response = await llm.acomplete("Hello, how are you?")
        print(f"LLM Response: {response.text[:100]}...")
        
        # Test chat
        from llama_index.core.llms import ChatMessage
        messages = [ChatMessage(role="user", content="What is 2+2?")]
        chat_response = await llm.achat(messages)
        print(f"Chat Response: {chat_response.message.content[:100]}...")
        
        print("✅ Ollama LLM test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Ollama LLM test failed: {e}")
        traceback.print_exc()
        return False


async def test_ollama_embeddings():
    """Test Ollama embeddings wrapper."""
    print("\nTesting Ollama Embeddings...")
    
    try:
        embedding = OllamaEmbedding(model="nomic-embed-text")
        
        # Test embedding generation
        text = "This is a test sentence."
        embedding_vector = await embedding._aget_text_embedding(text)
        
        print(f"Embedding dimension: {len(embedding_vector)}")
        print(f"First few values: {embedding_vector[:5]}")
        
        print("✅ Ollama Embeddings test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Ollama Embeddings test failed: {e}")
        traceback.print_exc()
        return False


def test_local_parsers():
    """Test local parsers."""
    print("\nTesting Local Parsers...")
    
    try:
        parser = LocalParser()
        
        # Test markdown parsing
        test_md = "# Test Document\n\nThis is a test markdown document.\n\n## Section 1\n\nSome content here."
        
        # Create a temporary markdown file
        with open("test_document.md", "w") as f:
            f.write(test_md)
        
        # Parse it
        parsed_doc = parser.parse("test_document.md")
        print(f"Parsed content: {parsed_doc.text[:100]}...")
        
        # Clean up
        os.remove("test_document.md")
        
        print("✅ Local Parsers test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Local Parsers test failed: {e}")
        return False


def test_local_extraction():
    """Test local extraction."""
    print("\nTesting Local Extraction...")
    
    try:
        extractor = LocalExtractor()
        
        # Test text extraction
        test_text = """
        This is an important document about machine learning.
        
        Key points:
        - Machine learning is a subset of AI
        - It involves training models on data
        - Deep learning is a type of ML
        
        The main benefits include automation and pattern recognition.
        """
        
        source = SourceText(text_content=test_text, filename="test.txt")
        extraction_result = extractor.extract(source)
        
        print(f"Extraction result keys: {list(extraction_result.data.keys())}")
        print(f"Summary: {extraction_result.data['summary'][:100]}...")
        print(f"Key points: {extraction_result.data['key_points'][:2]}")
        
        print("✅ Local Extraction test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Local Extraction test failed: {e}")
        return False


async def main():
    """Run all tests."""
    print("🧪 Testing Ollama Integration Components\n")
    
    # Check if Ollama is running
    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:11434/api/tags") as response:
                if response.status == 200:
                    print("✅ Ollama is running and accessible")
                else:
                    print("❌ Ollama is not responding properly")
                    return
    except Exception as e:
        print(f"❌ Cannot connect to Ollama: {e}")
        print("Please make sure Ollama is running on http://localhost:11434")
        return
    
    # Run tests
    tests = [
        test_ollama_llm(),
        test_ollama_embeddings(),
        test_local_parsers(),
        test_local_extraction(),
    ]
    
    results = []
    for test in tests:
        if asyncio.iscoroutine(test):
            result = await test
        else:
            result = test
        results.append(result)
    
    # Summary
    print(f"\n📊 Test Results: {sum(results)}/{len(results)} tests passed")
    
    if all(results):
        print("🎉 All tests passed! Ready to integrate with main codebase.")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")


if __name__ == "__main__":
    asyncio.run(main()) 
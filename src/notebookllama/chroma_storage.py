import os
import chromadb
from typing import List, Dict, Any, Optional
from llama_index.core import Document, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import CitationQueryEngine
from llama_index.core.embeddings import BaseEmbedding


class ChromaStorage:
    """ChromaDB storage wrapper for local vector storage."""
    
    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        collection_name: str = "notebookllama",
        embedding_model: Optional[BaseEmbedding] = None,
    ) -> None:
        """Initialize ChromaDB storage.
        
        Args:
            persist_directory: Directory to persist ChromaDB data
            collection_name: Name of the ChromaDB collection
            embedding_model: Embedding model to use
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
        except:
            self.collection = self.client.create_collection(name=collection_name)
        
        # Initialize vector store
        self.vector_store = ChromaVectorStore(chroma_collection=self.collection)
        
        # Initialize storage context
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store,
            embed_model=self.embedding_model
        )
        
        # Initialize index
        self.index = VectorStoreIndex.from_vector_store(
            self.vector_store,
            storage_context=self.storage_context,
            embed_model=self.embedding_model
        )
        
        # Initialize retriever
        self.retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=5,
        )
        
        # Initialize query engine
        self.query_engine = CitationQueryEngine(
            retriever=self.retriever,
            citation_chunk_size=256,
            citation_chunk_overlap=50,
        )
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the index."""
        self.index.insert_nodes(documents)
    
    def add_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add text to the index."""
        document = Document(text=text, metadata=metadata or {})
        self.add_documents([document])
    
    def query(self, query_text: str) -> str:
        """Query the index."""
        response = self.query_engine.query(query_text)
        return str(response)
    
    async def aquery(self, query_text: str) -> str:
        """Async query the index."""
        response = await self.query_engine.aquery(query_text)
        return str(response)
    
    def get_retriever(self):
        """Get the retriever."""
        return self.retriever
    
    def get_query_engine(self):
        """Get the query engine."""
        return self.query_engine
    
    def clear(self) -> None:
        """Clear all data from the collection."""
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.create_collection(name=self.collection_name)
        self.vector_store = ChromaVectorStore(chroma_collection=self.collection)
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store,
            embed_model=self.embedding_model
        )
        self.index = VectorStoreIndex.from_vector_store(
            self.vector_store,
            storage_context=self.storage_context,
            embed_model=self.embedding_model
        )
        self.retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=5,
        )
        self.query_engine = CitationQueryEngine(
            retriever=self.retriever,
            citation_chunk_size=256,
            citation_chunk_overlap=50,
        ) 
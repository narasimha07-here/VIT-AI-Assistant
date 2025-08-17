from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_experimental.text_splitter import SemanticChunker
from typing import List, Optional
from langchain.schema import Document
import logging

def create_embedding_model(model_name: str = "BAAI/bge-small-en-v1.5", **kwargs):
    """Create and validate embedding model"""
    try:
        model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=kwargs.get("model_kwargs", {"device": "cpu"}),
            encode_kwargs=kwargs.get("encode_kwargs", {"normalize_embeddings": True})
        )
        # Test the model with a small sample
        test_embedding = model.embed_query("test")
        if not isinstance(test_embedding, list) or len(test_embedding) == 0:
            raise ValueError("Embedding model failed to produce valid output")
        return model
    except Exception as e:
        logging.error(f"Failed to initialize embedding model: {str(e)}")
        raise

def create_vector_store(
    documents: List[Document],
    embedding_model: HuggingFaceEmbeddings,
    collection_name: str,
    persist_directory: str,
    top_k: int = 49,
    chunking_threshold: float = 0.5,
    chunking_method: str = "standard_deviation"
) -> Chroma:
    """
    Enhanced Chroma vector store creation with semantic chunking
    
    Args:
        documents: List of LangChain Document objects
        embedding_model: Initialized embedding model
        collection_name: Name for Chroma collection
        persist_directory: Directory to persist Chroma DB
        top_k: Number of chunks to retrieve
        chunking_threshold: Threshold for semantic chunking
        chunking_method: "standard_deviation" or "percentile"
    
    Returns:
        Configured retriever object
    """
    # Validate inputs
    if not documents or len(documents) == 0:
        raise ValueError("No documents provided for vector store creation")
    
    if not isinstance(documents[0], Document):
        raise TypeError("Input must be a list of LangChain Document objects")

    try:
        # -------------------------------
        # 1. Semantic Chunking with configurable parameters
        # -------------------------------
        text_splitter = SemanticChunker(
            embedding_model,
            breakpoint_threshold_type=chunking_method,
            breakpoint_threshold_amount=chunking_threshold
        )
        
        # Process documents in batches for large collections
        batch_size = 100
        chunked_docs = []
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            chunked_docs.extend(text_splitter.split_documents(batch))
            logging.info(f"Processed {min(i + batch_size, len(documents))}/{len(documents)} documents")

        # -------------------------------
        # 2. Create Chroma Vector Store with metadata handling
        # -------------------------------
        vector_store = Chroma.from_documents(
            documents=chunked_docs,
            embedding=embedding_model,
            collection_name=collection_name,
            persist_directory=persist_directory,
            collection_metadata={"hnsw:space": "cosine"}  # Optimize for cosine similarity
        )

        # -------------------------------
        # 3. Create configurable retriever
        # -------------------------------
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": top_k,
            }
        )

        logging.info(f"Created vector store with {len(chunked_docs)} chunks")
        return retriever

    except Exception as e:
        logging.error(f"Vector store creation failed: {str(e)}")
        raise
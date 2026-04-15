"""
retrieval — Document embedding and FAISS vector store package.

Exports the primary retrieval interface so callers can do:
    from retrieval import get_vectorstore, query_vectorstore
"""

from retrieval.vectorstore import get_vectorstore, query_vectorstore

__all__ = ["get_vectorstore", "query_vectorstore"]

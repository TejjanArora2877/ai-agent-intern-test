"""Abstract base interface for modular knowledge retrievers."""

from abc import ABC, abstractmethod
from typing import List, Optional

from src.models.schemas import RetrievedChunk


class BaseRetriever(ABC):
    """Abstract base class defining the retrieval contract for the agent."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 4, filter_active_only: bool = True) -> List[RetrievedChunk]:
        """
        Retrieve the top_k most relevant knowledge-base section chunks for a query.
        
        Args:
            query: The user query or enriched retrieval string.
            top_k: Maximum number of chunks to return.
            filter_active_only: Whether to strictly filter for active official documents.
            
        Returns:
            List of RetrievedChunk objects ordered by relevance score descending.
        """
        pass

    @abstractmethod
    def get_all_chunks(self) -> List[RetrievedChunk]:
        """Return all indexed section chunks."""
        pass

# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 Anant Patankar

"""
Core Data Models for RAG System

This module defines the essential data structures for the retrieval and generation 
pipeline. These models represent the interfaces between different system components.

Key structures:
- Chunk: The fundamental unit for indexing and retrieval
- SearchResult: Container for a retrieved chunk with relevance information
- QueryResult: Comprehensive result of a search operation
- GenerationResult: Output from the LLM generation process

These models form the "contract" between components, enabling clean handoffs
from retrieval to generation stages in the RAG pipeline.

Usage:
    # Create a chunk
    chunk = Chunk(
        content="The RAG system combines retrieval with generation...",
        doc_id="doc123",
        source="introduction.pdf"
    )

    # Use in search results
    search_result = SearchResult(chunk=chunk, score=0.92)
"""

from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field, asdict

from src.config.config import ContentType
from src.utils.helpers import generate_hash

@dataclass
class Chunk:
    """Chunk of text with metadata for vector database indexing."""
    content: str
    chunk_id: str = ""
    doc_id: str = ""
    source: str = ""
    section_id: Optional[str] = None
    section_title: str = ""
    parent_chunk_id: Optional[str] = None
    content_type: ContentType = ContentType.TEXT
    special_content_id: Optional[str] = None  # ID of table, image, formula, etc.
    strategy: str = ""
    start_pos: int = 0
    end_pos: int = 0
    embedding: Optional[List[float]] = None
    sparse_embedding: Optional[Dict[int, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.chunk_id:
            # Generate ID based on content and source
            id_text = f"{self.doc_id}|{self.section_id}|{self.content[:100]}"
            self.chunk_id = generate_hash(id_text)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            'chunk_id': self.chunk_id,
            'content': self.content,
            'doc_id': self.doc_id,
            'source': self.source,
            'section_id': self.section_id,
            'section_title': self.section_title,
            'parent_chunk_id': self.parent_chunk_id,
            'content_type': self.content_type.value,
            'special_content_id': self.special_content_id,
            'strategy': self.strategy,
            'start_pos': self.start_pos,
            'end_pos': self.end_pos,
            'metadata': self.metadata,
            'content_length': len(self.content) if self.content else 0
        }

        # Only include embeddings if they exist
        if self.embedding is not None:
            result['embedding'] = self.embedding
        if self.sparse_embedding is not None:
            result['sparse_embedding'] = self.sparse_embedding

        # Add all metadata fields at the top level for easier indexing
        for key, value in self.metadata.items():
            if key not in result:
                result[key] = value

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Chunk':
        """Create a Chunk from a dictionary."""
        # Handle content_type conversion from string to enum
        content_type = data.get('content_type', 'text')
        if isinstance(content_type, str):
            try:
                content_type = ContentType(content_type)
            except ValueError:
                content_type = ContentType.TEXT

        # Extract metadata
        metadata = data.get('metadata', {})

        # Create chunk
        chunk = cls(
            chunk_id=data.get('chunk_id', ''),
            content=data.get('content', ''),
            doc_id=data.get('doc_id', ''),
            source=data.get('source', ''),
            section_id=data.get('section_id'),
            section_title=data.get('section_title', ''),
            parent_chunk_id=data.get('parent_chunk_id'),
            content_type=content_type,
            special_content_id=data.get('special_content_id'),
            strategy=data.get('strategy', ''),
            start_pos=data.get('start_pos', 0),
            end_pos=data.get('end_pos', 0),
            embedding=data.get('embedding'),
            sparse_embedding=data.get('sparse_embedding'),
            metadata=metadata
        )
        return chunk

@dataclass
class SearchResult:
    """Container for search results."""
    chunk: Chunk
    score: float
    rank: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class QueryResult:
    """Container for RAG query results."""
    query: str
    answer: str
    context_chunks: List[Chunk]
    retrieved_chunks: List[SearchResult]  # All retrieved chunks before filtering
    execution_time: float
    model_name: str
    token_usage: Dict[str, int] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    # Add serialization helper method
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "query": self.query,
            "answer": self.answer,
            "execution_time": self.execution_time,
            "model_name": self.model_name,
            "token_usage": self.token_usage,
            "metadata": self.metadata,
            "error": self.error
        }

        # Handle context chunks (without embeddings)
        result["context_chunks"] = [
            {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "source": chunk.source,
                "section_id": chunk.section_id,
                "section_title": chunk.section_title,
                "content_type": chunk.content_type.value if hasattr(chunk.content_type, "value") else str(chunk.content_type),
                "content": chunk.content[:1000] + "..." if len(chunk.content) > 1000 else chunk.content
            }
            for chunk in self.context_chunks
        ]

        # Handle retrieved chunks (without embeddings)
        result["retrieved_chunks"] = [
            {
                "chunk_id": result.chunk.chunk_id if hasattr(result.chunk, "chunk_id") else "unknown",
                "score": result.score,
                "rank": result.rank
            }
            for result in self.retrieved_chunks
        ]

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QueryResult':
        """Create a QueryResult from a dictionary."""
        # Create a basic QueryResult with primitive types
        return cls(
            query=data.get("query", ""),
            answer=data.get("answer", ""),
            context_chunks=[],  # We can't fully reconstruct the chunks
            retrieved_chunks=[],  # We can't fully reconstruct the search results
            execution_time=data.get("execution_time", 0.0),
            model_name=data.get("model_name", ""),
            token_usage=data.get("token_usage", {}),
            metadata=data.get("metadata", {}),
            error=data.get("error")
        )

@dataclass
class GenerationResult:
    """Container for generation (LLM) results."""
    generated_text: str
    source_chunks: List[Chunk] = field(default_factory=list)
    query: str = ""
    prompt: str = ""
    model: str = ""
    temperature: float = 0.0
    execution_time: float = 0.0
    token_usage: Dict[str, int] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

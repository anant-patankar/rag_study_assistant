# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 Anant Patankar

"""
Configuration Module for RAG System

This module centralizes all configuration settings for the RAG system components:
    - Document Processing
    - Chunking
    - Embedding Generation
    - Vector Database
    - Generation (LLM)
    - Query Processing
    - System-wide settings
"""

import os
import json
import torch
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from enum import Enum


class ContentType(Enum):
    """Enum for different content types."""
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    FORMULA = "formula"
    CODE = "code"
    LIST = "list"
    HEADING = "heading"
    REFERENCE = "reference"
    METADATA = "metadata"

@dataclass
class ProcessingConfig:
    """Configuration for document processing."""
    extract_tables: bool = True
    extract_images: bool = True
    extract_formulas: bool = True
    extract_code_blocks: bool = True
    max_image_size: int = 10485760  # 10MB
    image_extraction_method: str = "automatic"  # automatic, ocr, or none
    table_extraction_method: str = "automatic"  # automatic, heuristic, or model-based
    formula_extraction_method: str = "pattern"  # pattern or model-based
    code_extraction_method: str = "pattern"  # pattern or model-based
    preserve_whitespace: bool = False
    detect_languages: bool = True
    extract_metadata: bool = True
    extract_sections: bool = True
    extract_references: bool = True
    extract_links: bool = True
    clean_header_footer: bool = True
    min_section_length: int = 50
    max_header_footer_fraction: float = 0.1
    ocr_enabled: bool = False
    ocr_language: str = "eng"
    ocr_threshold: float = 0.5
    custom_content_extractors: Dict[str, Callable] = field(default_factory=dict)
    custom_processors: Dict[str, Callable] = field(default_factory=dict)

@dataclass
class ChunkingConfig:
    """Configuration for text chunking."""
    strategy: str = "hybrid"  # fixed, semantic, hybrid, or document-specific
    chunk_size: int = 1000
    chunk_overlap: int = 200
    respect_sentence_boundaries: bool = True
    respect_paragraph_boundaries: bool = True
    respect_section_boundaries: bool = True
    respect_document_boundaries: bool = True
    chunk_document_heading: bool = True
    include_metadata: bool = True
    include_breadcrumbs: bool = True
    include_section_path: bool = True
    include_special_chunks: bool = True  # for tables, images, formulas, etc.
    include_table_summaries: bool = True
    include_image_captions: bool = True
    max_chunks_per_doc: Optional[int] = None
    custom_chunking_strategies: Dict[str, Callable] = field(default_factory=dict)

@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""
    model_name: str = "all-MiniLM-L6-v2"
    batch_size: int = 32
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    normalize_embeddings: bool = True
    pooling_strategy: str = "mean"  # mean, max, or cls
    cache_embeddings: bool = True
    embeddings_cache_path: str = "./artifacts/embeddings_cache"
    use_domain_adaptation: bool = False
    domain_adaptation_model: Optional[str] = None
    custom_embedding_functions: Dict[str, Callable] = field(default_factory=dict)
    content_type_embeddings: bool = True  # Use specialized embeddings for different content types
    hybrid_embeddings: bool = False  # Combine dense and sparse embeddings
    sparse_embeddings_ratio: float = 0.3  # Only used if hybrid_embeddings is True
    embedding_dim: int = 384  # Will be overridden based on model
    cross_encoder_reranking: bool = False
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    content_type_prefixes: bool = True  # Add content type prefixes to improve differentiation
    embedding_batch_size: int = 32  # Batch size for parallel embedding generation
    cache_format: str = "pickle"  # pickle or mmap
    parallel_encoding: bool = True  # Use parallel encoding for faster processing

@dataclass
class VectorDBConfig:
    """Configuration for vector database interaction."""
    db_type: str = "faiss"  # faiss, hnswlib, or custom
    index_type: str = "flat"  # flat, hnsw, ivf, ivfpq, etc.
    metric_type: str = "cosine"  # cosine, l2, dot, etc.
    dimension: int = 384  # Will be set based on embedding model
    store_vectors: bool = True
    store_content: bool = True
    store_metadata: bool = True
    num_partitions: int = 100  # For IVF-based indices
    num_probes: int = 10  # For IVF-based indices
    ef_construction: int = 200  # For HNSW-based indices
    ef_search: int = 50  # For HNSW-based indices
    m_parameter: int = 16  # For HNSW-based indices
    pq_bytes: int = 8  # For PQ-based indices
    use_multiple_indices: bool = False  # Use different indices for different content types
    replication_factor: int = 1
    shards: int = 1
    custom_index_builder: Optional[Callable] = None
    vector_db_path: str = "./artifacts/vector_store"
    create_sql_metadata_storage: bool = True
    sql_connection_string: str = "sqlite:///metadata.db"
    auto_save: bool = True
    save_interval: int = 300  # seconds
    backup_before_save: bool = True
    max_backups: int = 3
    max_index_size: int = 1000000  # Maximum vectors in a single index
    hybrid_search_weights: Dict[str, float] = field(default_factory=lambda: {"vector": 0.7, "sparse": 0.3})
    enable_batched_updates: bool = True  # For better performance with large inserts
    batch_size: int = 1000  # Size of batches for updates
    enable_versioning: bool = True  # Track versions of the index
    version_tag: str = ""  # Custom version tag
    auto_reindex_threshold: float = 0.2  # Auto-reindex when 20% of vectors are deleted

@dataclass
class RetrievalConfig:
    """Configuration for retrieval strategies."""
    top_k: int = 5
    minimum_score: float = 0.5
    use_hybrid_search: bool = True
    hybrid_search_weights: Dict[str, float] = field(default_factory=lambda: {"vector": 0.7, "bm25": 0.3})
    use_metadata_filtering: bool = True
    use_reranking: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_k: int = 100
    use_mmr: bool = True  # Maximum Marginal Relevance for diversity
    mmr_lambda: float = 0.5
    expand_keywords: bool = True
    query_expansion_model: Optional[str] = None
    query_transformations: List[str] = field(default_factory=lambda: ["original", "expanded"])
    query_by_document: bool = False
    custom_retrievers: Dict[str, Callable] = field(default_factory=dict)
    filter_by_source: bool = True
    filter_by_content_type: bool = True
    filter_by_metadata: bool = True
    use_query_decomposition: bool = True

@dataclass
class EvaluationConfig:
    """Configuration for RAG system evaluation."""
    test_set_path: Optional[str] = None
    eval_metrics: List[str] = field(default_factory=lambda: ["precision", "recall", "f1", "ndcg", "mrr"])
    relevance_threshold: float = 0.5
    num_eval_queries: int = 100
    max_retrievals_per_query: int = 10
    use_human_evaluation: bool = False
    log_metrics: bool = True
    log_path: str = "./evaluation_logs"
    baseline_systems: List[str] = field(default_factory=list)
    test_variations: List[Dict[str, Any]] = field(default_factory=list)
    use_crossvalidation: bool = False
    crossvalidation_folds: int = 5
    custom_evaluators: Dict[str, Callable] = field(default_factory=dict)

@dataclass
class GenerationConfig:
    """Configuration for LLM generation using Ollama."""
    model_name: str = "deepseek-r1:1.5b"  # Name of the Ollama model to use
    ollama_base_url: str = "http://localhost:11434"  # Base URL for Ollama API
    temperature: float = 0.7  # Temperature for generation
    top_p: float = 0.9  # Top-p sampling parameter
    top_k: int = 40  # Top-k sampling parameter
    max_tokens: int = 1000  # Maximum number of tokens to generate
    system_prompt: str = "You are a helpful assistant. Use the provided context to answer the question."
    prompt_template: str = """
    Context information is below.
    ---------------------
    {context}
    ---------------------
    Given the context information and not prior knowledge, answer the question: {query}
    """
    include_sources: bool = True  # Whether to include source information in the response
    streaming: bool = False  # Whether to stream the response or wait for complete generation
    num_context_chunks: int = 5  # Number of context chunks to include in the prompt
    timeout: int = 60  # Timeout for Ollama API requests in seconds

@dataclass
class QueryConfig:
    """Configuration for query processing."""
    use_hybrid_search: bool = False  # Use hybrid search (dense + sparse)
    rerank_results: bool = True  # Rerank results using cross-encoder
    expand_queries: bool = True  # Expand queries for better retrieval
    filter_by_source: bool = False  # Filter results by source
    filter_by_content_type: bool = False  # Filter results by content type
    filter_by_metadata: Dict[str, Any] = field(default_factory=dict)  # Custom metadata filters
    max_retrieved_chunks: int = 10  # Maximum number of chunks to retrieve
    min_relevance_score: float = 0.6  # Minimum relevance score for chunks
    use_mmr: bool = True  # Use Maximum Marginal Relevance for diversity
    mmr_lambda: float = 0.5  # Lambda parameter for MMR

@dataclass
class RagSystemConfig:
    """Overall configuration for the RAG system."""
    processing_config: ProcessingConfig = field(default_factory=ProcessingConfig)
    chunking_config: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding_config: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_db_config: VectorDBConfig = field(default_factory=VectorDBConfig)
    generation_config: GenerationConfig = field(default_factory=GenerationConfig)
    query_config: QueryConfig = field(default_factory=QueryConfig)
    evaluation_config: EvaluationConfig = field(default_factory=EvaluationConfig)
    system_name: str = "RAG System"
    system_version: str = "1.0.0"
    cache_dir: str = "./artifacts/rag_cache"
    log_level: str = "INFO"
    enable_telemetry: bool = False
    enable_caching: bool = True
    num_threads: int = 4  # Number of threads for parallel processing
    save_config_with_results: bool = True  # From original SystemConfig
    custom_components: Dict[str, Any] = field(default_factory=dict)  # From original SystemConfig

    def to_dict(self) -> Dict[str, Any]:
        """Convert system config to dictionary, handling nested dataclasses."""
        return {
            "processing_config": asdict(self.processing_config),
            "chunking_config": asdict(self.chunking_config),
            "embedding_config": asdict(self.embedding_config),
            "vector_db_config": asdict(self.vector_db_config),
            "generation_config": asdict(self.generation_config),
            "query_config": asdict(self.query_config),
            "evaluation_config": asdict(self.evaluation_config),
            "log_level": self.log_level,
            "num_threads": self.num_threads,
            "system_name": self.system_name,
            "system_version": self.system_version,
            "cache_dir": self.cache_dir,
            "enable_telemetry": self.enable_telemetry,
            "enable_caching": self.enable_caching,
            "save_config_with_results": self.save_config_with_results,
            "custom_components": self.custom_components
        }

    def save_to_file(self, filepath: str) -> None:
        """Save configuration to a JSON file."""
        # Get the dictionary representation
        config_dict = self.to_dict()

        # Remove callable objects which can't be serialized
        for config_section in config_dict.values():
            if isinstance(config_section, dict):
                keys_to_remove = []
                for key, value in config_section.items():
                    if callable(value) or key.startswith('_'):
                        keys_to_remove.append(key)
                
                for key in keys_to_remove:
                    config_section.pop(key, None)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2)

    @classmethod
    def load_from_file(cls, filepath: str) -> 'RagSystemConfig':
        """Load configuration from a JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)

        # Create component configs
        processing_config = ProcessingConfig(**config_dict.get("processing_config", {}))
        chunking_config = ChunkingConfig(**config_dict.get("chunking_config", {}))
        embedding_config = EmbeddingConfig(**config_dict.get("embedding_config", {}))
        vector_db_config = VectorDBConfig(**config_dict.get("vector_db_config", {}))
        generation_config = GenerationConfig(**config_dict.get("generation_config", {}))
        query_config = QueryConfig(**config_dict.get("query_config", {}))
        evaluation_config = EvaluationConfig(**config_dict.get("evaluation_config", {}))

        # Create main config
        return cls(
            processing_config=processing_config,
            chunking_config=chunking_config,
            embedding_config=embedding_config,
            vector_db_config=vector_db_config,
            generation_config=generation_config,
            query_config=query_config,
            evaluation_config=evaluation_config,
            system_name=config_dict.get("system_name", "RAG System"),
            system_version=config_dict.get("system_version", "1.0.0"),
            cache_dir=config_dict.get("cache_dir", "./artifacts/rag_cache"),
            log_level=config_dict.get("log_level", "INFO"),
            enable_telemetry=config_dict.get("enable_telemetry", False),
            enable_caching=config_dict.get("enable_caching", True),
            num_threads=config_dict.get("num_threads", 4),
            save_config_with_results=config_dict.get("save_config_with_results", True),
            custom_components=config_dict.get("custom_components", {})
        )

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'RagSystemConfig':
        """Create a SystemConfig from a dictionary."""
        processing_config = ProcessingConfig(**config_dict.get("processing_config", {}))
        chunking_config = ChunkingConfig(**config_dict.get("chunking_config", {}))
        embedding_config = EmbeddingConfig(**config_dict.get("embedding_config", {}))
        vector_db_config = VectorDBConfig(**config_dict.get("vector_db_config", {}))
        evaluation_config = EvaluationConfig(**config_dict.get("evaluation_config", {}))
        generation_config = GenerationConfig(**config_dict.get("generation_config", {}))
        query_config = QueryConfig(**config_dict.get("query_config", {}))

        return cls(
            processing_config=processing_config,
            chunking_config=chunking_config,
            embedding_config=embedding_config,
            vector_db_config=vector_db_config,
            evaluation_config=evaluation_config,
            generation_config=generation_config,
            query_config=query_config,
            log_level=config_dict.get("log_level", "INFO"),
            num_threads=config_dict.get("num_threads", max(1, os.cpu_count() // 2)),
            system_name=config_dict.get("system_name", "RAG System"),
            system_version=config_dict.get("system_version", "1.0.0"),
            cache_dir=config_dict.get("cache_dir", "./cache"),
            enable_telemetry=config_dict.get("enable_telemetry", False),
            enable_caching=config_dict.get("enable_caching", True),
            save_config_with_results=config_dict.get("save_config_with_results", True),
            custom_components=config_dict.get("custom_components", {})
        )

    def save(self, file_path: str) -> None:
        """Save system configuration to a JSON file."""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, file_path: str) -> 'RagSystemConfig':
        """Load system configuration from a JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

# Default configurations for different use cases
def get_default_config() -> RagSystemConfig:
    """Get default configuration for general use."""
    return RagSystemConfig()

def get_lightweight_config() -> RagSystemConfig:
    """Get lightweight configuration for resource-constrained environments."""
    config = RagSystemConfig()
    config.embedding_config.model_name = "all-MiniLM-L6-v2"  # Small but effective model
    config.embedding_config.batch_size = 8
    config.embedding_config.parallel_encoding = False
    config.vector_db_config.index_type = "flat"  # Simplest index type
    config.chunking_config.chunk_size = 512  # Smaller chunks
    config.generation_config.model_name = "deepseek-r1:1.5b"  # A smaller Ollama model
    config.num_threads = 2
    return config

def get_high_accuracy_config() -> RagSystemConfig:
    """Get configuration optimized for accuracy."""
    config = RagSystemConfig()
    config.embedding_config.model_name = "all-mpnet-base-v2"  # More accurate embedding model
    config.embedding_config.normalize_embeddings = True
    config.chunking_config.strategy = "semantic"
    config.chunking_config.chunk_size = 800
    config.chunking_config.chunk_overlap = 200
    config.vector_db_config.index_type = "hnsw"
    config.vector_db_config.ef_search = 200  # Higher precision search
    config.query_config.use_hybrid_search = True
    config.query_config.rerank_results = True
    config.query_config.max_retrieved_chunks = 15  # Retrieve more candidates
    config.generation_config.temperature = 0.3  # Lower temperature for more focused responses
    return config

def get_high_performance_config() -> RagSystemConfig:
    """Get configuration optimized for performance."""
    config = RagSystemConfig()
    config.embedding_config.model_name = "all-MiniLM-L6-v2"  # Fast embedding model
    config.embedding_config.batch_size = 64
    config.embedding_config.parallel_encoding = True
    config.chunking_config.strategy = "fixed"  # Simplest chunking strategy
    config.vector_db_config.index_type = "ivfpq"  # More compressed index format
    config.vector_db_config.enable_batched_updates = True
    config.vector_db_config.batch_size = 2000
    config.query_config.use_mmr = False  # Skip diversity calculation for speed
    config.query_config.max_retrieved_chunks = 5  # Fewer chunks for faster generation
    config.generation_config.max_tokens = 800
    config.num_threads = 8  # Use more threads
    return config


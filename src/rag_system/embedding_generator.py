# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 Anant Patankar

"""
Embedding Generator Module for RAG System

This module handles the generation of vector embeddings for document chunks, 
addressing the critical "representation" component of RAG systems.

What this thing actually does:
- Converts text chunks into dense vector embeddings using SentenceTransformers
- Handles caching to avoid recomputing embeddings (major time-saver!)
- Supports content-specific embedding strategies (different handling for tables vs code)
- Implements hybrid search with both dense and sparse vectors when needed
- Handles batching + optimizations to prevent your GPU from melting

Key Features:
- Smart caching with thread-safe operations
- Batched processing to handle large document collections efficiently
- Content type specialization with prefixing for better retrieval
- Domain adaptation support for fine-tuned embedding models
- Parallel encoding options with configurable batch sizes

Performance Notes:
- First-time embedding is slow-ish, but cached retrievals are lightning fast
- Memory usage scales with batch size, so adjust according to your hardware
- For >100k documents, consider using smaller embedding models or sharding

Common Issues:
- GPU OOM errors: Reduce batch size or switch to CPU if needed
- Slow cold start: Expected on first run, gets faster with caching
- Disk usage growth: Cache directory can get large, implement pruning if needed

Practical Example:
config = EmbeddingConfig(
    model_name="all-MiniLM-L6-v2",  # Decent balance of speed/quality
    cache_embeddings=True,  # Always true in prod, your future self will thank you
    device="cuda"  # Use CPU if no GPU available
)
embedder = EmbeddingGenerator(config)
chunks_with_embeddings = embedder.generate_embeddings(document_chunks)

Developed by: Anant Patankar
Last major update: March 27, 2025
"""
# where there is "if configured" in comment that option can be configured from configuration
import os
import json
import logging
import numpy as np
import pickle
import hashlib
import threading
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum

# Vector models
import torch
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config.data_models import ContentType, Chunk

import sys
import os

from src.utils.helpers import Timer, batch_items, normalize_vector, ensure_directory

# Configure logging
logger = logging.getLogger('rag_system.embeddings')

from src.config.config import EmbeddingConfig

class EmbeddingGenerator:
    """Class for generating embeddings for document chunks."""

    def __init__(self, config: EmbeddingConfig):
        """
        Set up the embedding generator with your config.

        Args:
            config: All the settings for how embeddings should be created
                (model, batch size, cache settings, etc)
        """
        self.config = config

        # Initialize embedding model
        logger.info(f"Loading embedding model: {config.model_name}")
        self.embedding_model = SentenceTransformer(config.model_name, device=config.device)

        # Update embedding dimension based on actual model
        self.config.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.config.embedding_dim}")

        # Initialize cross-encoder model if reranking is enabled
        self.cross_encoder = None
        if config.cross_encoder_reranking:
            try:
                from sentence_transformers import CrossEncoder
                logger.info(f"Loading cross-encoder model: {config.cross_encoder_model}")
                self.cross_encoder = CrossEncoder(config.cross_encoder_model, device=config.device)
            except Exception as e:
                logger.warning(f"Failed to load cross-encoder: {e}. Reranking will be disabled.")

        # Initialize sparse embeddings (for hybrid search) if configured
        self.sparse_vectorizer = None
        if config.hybrid_embeddings:
            logger.info("Initializing sparse vectorizer for hybrid embeddings")
            self.sparse_vectorizer = TfidfVectorizer(
                lowercase=True,
                stop_words='english',
                ngram_range=(1, 2),
                max_features=50000
            )

        # Initialize domain adaptation if configured
        self.domain_adapter = None
        if config.use_domain_adaptation and config.domain_adaptation_model:
            logger.info(f"Loading domain adaptation model: {config.domain_adaptation_model}")
            try:
                # This could be a fine-tuned model specific to your domain
                self.domain_adapter = SentenceTransformer(config.domain_adaptation_model, device=config.device)
            except Exception as e:
                logger.warning(f"Failed to load domain adaptation model: {e}. Domain adaptation disabled.")

        # Initialize cache if enabled
        self._cache_lock = threading.Lock()
        if config.cache_embeddings:
            ensure_directory(config.embeddings_cache_path)

        # Define content type prefixes for specialized embeddings
        self.content_prefixes = {
            ContentType.TEXT: "TEXT: ",
            ContentType.TABLE: "TABLE: ",
            ContentType.IMAGE: "IMAGE: ",
            ContentType.FORMULA: "FORMULA: ",
            ContentType.CODE: "CODE: ",
            ContentType.LIST: "LIST: ",
            ContentType.HEADING: "HEADING: ",
            ContentType.REFERENCE: "REFERENCE: ",
            ContentType.METADATA: "METADATA: "
        }

        # Register custom embedding functions
        self.custom_embedding_funcs = {}
        for content_type_name, func in config.custom_embedding_functions.items():
            try:
                content_type = ContentType(content_type_name)
                self.custom_embedding_funcs[content_type] = func
            except ValueError:
                logger.warning(f"Unknown content type: {content_type_name}")

    def generate_embeddings(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Generate embeddings for a list of chunks.
        I implemented this for handling caching, batching, and content-type
        specific embedding strategies for optimal retrieval.
        Args:
            chunks: List of document chunks

        Returns:
            List of chunks with embeddings
        """
        with Timer("Generate Embeddings") as timer:
            logger.info(f"Generating embeddings for {len(chunks)} chunks")

            # Group chunks by content type if specialized embeddings are enabled
            if self.config.content_type_embeddings:
                chunks_by_type = {}
                for chunk in chunks:
                    if chunk.content_type not in chunks_by_type:
                        chunks_by_type[chunk.content_type] = []
                    chunks_by_type[chunk.content_type].append(chunk)

                # Process each content type separately
                processed_chunks = []
                for content_type, content_chunks in chunks_by_type.items():
                    with_embeddings = self._process_chunks_by_type(content_chunks, content_type)
                    processed_chunks.extend(with_embeddings)

                # Sort chunks back to original order if needed
                chunk_dict = {chunk.chunk_id: chunk for chunk in processed_chunks}
                result_chunks = [chunk_dict[chunk.chunk_id] for chunk in chunks]
            else:
                # Process all chunks together
                result_chunks = self._process_chunks_by_type(chunks)

            # Generate sparse embeddings if hybrid search is enabled
            if self.config.hybrid_embeddings and self.sparse_vectorizer is not None:
                self._generate_sparse_embeddings(result_chunks)

        processing_time = timer.elapsed
        logger.info(f"Embedding generation completed in {processing_time:.2f}s")
        return result_chunks

    def _process_chunks_by_type(self, chunks: List[Chunk], 
                            content_type: Optional[ContentType] = None) -> List[Chunk]:
        """
            Handle chunks of a specific content type - either with custom logic or standard embedding.
        """
        # Check if there's a custom embedding function for this content type
        if content_type in self.custom_embedding_funcs:
            return self._custom_embedding_generation(chunks, content_type)

        # Check cache first if enabled
        if self.config.cache_embeddings:
            chunks_to_embed = []
            cached_chunks = []

            for chunk in chunks:
                cache_path = self._get_cache_path(chunk)
                if cache_path.exists():
                    # Load from cache
                    try:
                        cached_embedding = self._load_from_cache(cache_path)
                        chunk.embedding = cached_embedding
                        cached_chunks.append(chunk)
                    except Exception as e:
                        logger.warning(f"Failed to load from cache: {e}")
                        chunks_to_embed.append(chunk)
                else:
                    chunks_to_embed.append(chunk)

            logger.info(f"Loaded {len(cached_chunks)} embeddings from cache, "
                    f"generating {len(chunks_to_embed)} new embeddings")

            # Generate embeddings for non-cached chunks
            if chunks_to_embed:
                embedded_chunks = self._generate_dense_embeddings(chunks_to_embed, content_type)

                # Save to cache
                if self.config.cache_embeddings:
                    for chunk in embedded_chunks:
                        cache_path = self._get_cache_path(chunk)
                        try:
                            self._save_to_cache(chunk.embedding, cache_path)
                        except Exception as e:
                            logger.warning(f"Failed to save to cache: {e}")

                return cached_chunks + embedded_chunks
            else:
                return cached_chunks
        else:
            # No caching, generate all embeddings
            return self._generate_dense_embeddings(chunks, content_type)

    def _generate_dense_embeddings(self, chunks: List[Chunk], 
                                content_type: Optional[ContentType] = None) -> List[Chunk]:
        """
            Generating dense embeddings for chunks.
        """
        if not chunks:
            return []

        # Prepare texts for embedding with appropriate prefixes
        texts = []
        for chunk in chunks:
            text = chunk.content

            # Add content type prefix if configured
            if self.config.content_type_prefixes and content_type is not None:
                prefix = self.content_prefixes.get(content_type, "")
                text = prefix + text

            texts.append(text)

        # Embeddings generation in batches
        all_embeddings = []
        batch_size = self.config.embedding_batch_size

        batched_texts = batch_items(texts, batch_size)
        for batch_texts in batched_texts:

            # Embeddings generation for this batch
            if self.config.parallel_encoding:
                batch_embeddings = self.embedding_model.encode(
                    batch_texts,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=self.config.normalize_embeddings
                )
            else:
                # Sequential processing (works slowly but less memory intensive)
                batch_embeddings = []
                for text in batch_texts:
                    emb = self.embedding_model.encode(
                        text,
                        convert_to_numpy=True,
                        normalize_embeddings=self.config.normalize_embeddings
                    )
                    batch_embeddings.append(emb)
                batch_embeddings = np.array(batch_embeddings)

            all_embeddings.append(batch_embeddings)

        # Combining all batches
        embeddings = np.vstack(all_embeddings) if len(all_embeddings) > 1 else all_embeddings[0]

        # Apply domain adaptation if configured
        if self.config.use_domain_adaptation and self.domain_adapter is not None:
            adapted_embeddings = []
            batched_texts = batch_items(texts, batch_size)
            for batch_texts in batched_texts:

                # Get domain-specific embeddings
                domain_embeddings = self.domain_adapter.encode(
                    batch_texts,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=self.config.normalize_embeddings
                )

                adapted_embeddings.append(domain_embeddings)

            domain_embeddings = np.vstack(adapted_embeddings) if len(adapted_embeddings) > 1 else adapted_embeddings[0]

            # Combine general and domain-specific embeddings
            # This is a simple averaging approach - more sophisticated methods could be used
            embeddings = (embeddings + domain_embeddings) / 2

            # Normalize if provided in configuration
            if self.config.normalize_embeddings:
                embeddings = np.vstack([normalize_vector(emb) for emb in embeddings])

        # Assign embeddings to chunks
        for i, chunk in enumerate(chunks):
            chunk.embedding = embeddings[i].tolist()

        return chunks

    def _generate_sparse_embeddings(self, chunks: List[Chunk]) -> None:
        """
            Creates TF-IDF sparse vectors for chunks when we need hybrid search capabilities.
        """
        if not self.sparse_vectorizer or not chunks:
            return

        # Fit vectorizer if not fitted already
        texts = [chunk.content for chunk in chunks]

        if not hasattr(self.sparse_vectorizer, 'vocabulary_'):
            self.sparse_vectorizer.fit(texts)

        # Transform texts to sparse vectors
        sparse_matrix = self.sparse_vectorizer.transform(texts)

        # Convert to dictionary format for each chunk
        for i, chunk in enumerate(chunks):
            # Get the sparse vector for this chunk
            sparse_vector = sparse_matrix[i]

            # Convert to dictionary format {index: value}
            indices = sparse_vector.indices
            values = sparse_vector.data

            # Create a dictionary of non-zero elements
            sparse_dict = {int(idx): float(val) for idx, val in zip(indices, values)}

            # Add to chunk
            chunk.sparse_embedding = sparse_dict

    def _custom_embedding_generation(self, chunks: List[Chunk], 
                                content_type: ContentType) -> List[Chunk]:
        """
            custom function for generating embeddings for specific content type.
        """
        custom_func = self.custom_embedding_funcs[content_type]

        try:
            # Custom function should accept a list of chunks and return the same list with embeddings
            return custom_func(chunks, self.embedding_model, self.config)
        except Exception as e:
            logger.error(f"Error in custom embedding function for {content_type}: {e}")
            # Fall back to standard embedding
            return self._generate_dense_embeddings(chunks, content_type)

    def _get_cache_path(self, chunk: Chunk) -> Path:
        # Create a unique hash for the chunk content and model
        content_hash = hashlib.md5(chunk.content.encode()).hexdigest()
        model_hash = hashlib.md5(self.config.model_name.encode()).hexdigest()

        # Create a unique cache key
        cache_key = f"{content_hash}_{model_hash}"
        if self.config.content_type_prefixes:
            cache_key += f"_{chunk.content_type.value}"

        cache_dir = Path(self.config.embeddings_cache_path)
        return cache_dir / f"{cache_key}.{self.config.cache_format}"

    def _save_to_cache(self, embedding: List[float], cache_path: Path) -> None:
        with self._cache_lock:
            if self.config.cache_format == 'pickle':
                with open(cache_path, 'wb') as f:
                    pickle.dump(embedding, f)
            else:
                # Save as numpy mmap for very large embeddings
                np.save(cache_path, np.array(embedding))

    def _load_from_cache(self, cache_path: Path) -> List[float]:
        if self.config.cache_format == 'pickle':
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        else:
            # Load from numpy mmap
            return np.load(cache_path).tolist()

    def encode_query(self, query: str, content_type: Optional[ContentType] = None) -> np.ndarray:
        """
        encode a query string into an embedding vector.

        Args:
            query: The query string
            content_type: Optional content type for specialized encoding

        Returns:
            Query embedding as numpy array
        """
        # Add content type prefix if configured
        if self.config.content_type_prefixes and content_type is not None:
            prefix = self.content_prefixes.get(content_type, "")
            query = prefix + query

        query_embedding = self.embedding_model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize_embeddings
        )

        # apply domain adaptation if configured
        if self.config.use_domain_adaptation and self.domain_adapter is not None:
            domain_embedding = self.domain_adapter.encode(
                query,
                convert_to_numpy=True,
                normalize_embeddings=self.config.normalize_embeddings
            )

            # combine general and domain-specific embeddings
            query_embedding = (query_embedding + domain_embedding) / 2

            # Normalize if configured
            if self.config.normalize_embeddings:
                query_embedding = query_embedding / np.linalg.norm(query_embedding)

        return query_embedding

    def encode_queries(self, queries: List[str], 
                        content_type: Optional[ContentType] = None) -> np.ndarray:
        """
        Encode multiple query strings into embedding vectors.

        Args:
            queries: List of query strings
            content_type: Optional content type for specialized encoding

        Returns:
            Query embeddings as numpy array
        """
        # Add content type prefix if configured
        if self.config.content_type_prefixes and content_type is not None:
            prefix = self.content_prefixes.get(content_type, "")
            queries = [prefix + q for q in queries]

        # embeddings generate
        query_embeddings = self.embedding_model.encode(
            queries,
            batch_size=self.config.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize_embeddings
        )

        # Apply domain adaptation if configured
        if self.config.use_domain_adaptation and self.domain_adapter is not None:
            domain_embeddings = self.domain_adapter.encode(
                queries,
                batch_size=self.config.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=self.config.normalize_embeddings
            )

            # Combine general and domain-specific embeddings
            query_embeddings = (query_embeddings + domain_embeddings) / 2

            # Normalize if configured
            if self.config.normalize_embeddings:
                norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
                query_embeddings = query_embeddings / norms

        return query_embeddings

    def encode_sparse_query(self, query: str) -> Dict[int, float]:
        """
        Encode a query into a sparse vector for hybrid search.

        Args:
            query: The query string

        Returns:
            Sparse vector as dictionary {index: value}
        """
        if not self.sparse_vectorizer or not hasattr(self.sparse_vectorizer, 'vocabulary_'):
            logger.error("Sparse vectorizer not initialized or fitted")
            return {}

        # Transform query to sparse vector
        sparse_vector = self.sparse_vectorizer.transform([query])

        # Convert to dictionary format
        indices = sparse_vector.indices
        values = sparse_vector.data

        return {int(idx): float(val) for idx, val in zip(indices, values)}

    def rerank(self, query: str, chunks: List[Chunk], top_k: int = 10) -> List[Tuple[Chunk, float]]:
        """
        Rerank retrieved chunks using cross-encoder if available.

        Args:
            query: The query string
            chunks: List of retrieved chunks
            top_k: Number of top results to return

        Returns:
            List of (chunk, score) tuples, sorted by descending score
        """
        if not self.cross_encoder or not chunks:
            return [(chunk, 0.0) for chunk in chunks[:top_k]]

        # Prepare input pairs
        pairs = [(query, chunk.content) for chunk in chunks]

        # Score all the pairs
        scores = self.cross_encoder.predict(pairs)

        # Create (chunk, score) pairs and sort by score
        chunk_score_pairs = list(zip(chunks, scores))
        ranked_pairs = sorted(chunk_score_pairs, key=lambda x: x[1], reverse=True)

        # Return top_k results
        return ranked_pairs[:top_k]

    def generate_embeddings_file(self, chunks: List[Chunk], output_path: str) -> None:
        """
        Generate embeddings for chunks and save to file for later use.

        Args:
            chunks: List of document chunks
            output_path: Path to save embeddings
        """
        # Generate embeddings
        chunks_with_embeddings = self.generate_embeddings(chunks)

        # Extract embeddings and metadata
        embeddings_data = []
        for chunk in chunks_with_embeddings:
            data = {
                'chunk_id': chunk.chunk_id,
                'doc_id': chunk.doc_id,
                'source': chunk.source,
                'section_id': chunk.section_id,
                'section_title': chunk.section_title,
                'content_type': chunk.content_type.value,
                'embedding': chunk.embedding
            }

            # Add sparse embedding if available
            if chunk.sparse_embedding:
                data['sparse_embedding'] = chunk.sparse_embedding

            embeddings_data.append(data)

        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(embeddings_data, f)

        logger.info(f"Saved embeddings for {len(chunks)} chunks to {output_path}")

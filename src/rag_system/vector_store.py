# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 Anant Patankar

"""
Vector Store

This module handles all the vector database operations that power semantic search:
    - Stores document chunk embeddings in vector indices
    - Supports multiple backends (FAISS, HNSW) with smart config
    - Enables lightning-fast similarity search with filtering
    - Handles hybrid search (dense + sparse vectors)
    - Manages metadata storage alongside vectors

It's the database engine of the RAG system - chunk vectors go in,
relevant results come out. Lots of performance optimizations under 
the hood like batched updates, auto-rebuilding, and thread safety.

Usage:
> config = VectorDBConfig(db_type="faiss", index_type="hnsw")
> vector_store = VectorStore(config, embedding_generator)
> vector_store.add_chunks(chunks_with_embeddings)
> results = vector_store.search("my query", top_k=5)

Warning: Memory usage scales with your vector count, so monitor this
on large collections. The auto-save feature is your friend.
"""

import os
import json
import logging
import numpy as np
import time
import threading
import sqlite3
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
import shutil
import pickle

# Vector libraries
import faiss
import hnswlib

from src.config.data_models import ContentType, Chunk

from src.rag_system.embedding_generator import EmbeddingGenerator, EmbeddingConfig
from src.utils.helpers import safe_json_loads, ensure_directory, ThreadSafeCounter

from src.config.config import VectorDBConfig
from src.config.data_models import SearchResult

# Configure logging
logger = logging.getLogger('rag_system.vector_store')


class VectorStore:
    """
    Vector database for storing and retrieving document chunks.

    This class provides an abstraction over different vector database backends,
    handling the storage and retrieval of document chunks based on vector similarity.
    """
    def __init__(self, config: VectorDBConfig, embedding_generator: EmbeddingGenerator = None):
        """
        Initialize the vector store.
        """
        self.config = config
        self.embedding_generator = embedding_generator

        # Ensure vector dimension matches embedding model
        if embedding_generator:
            self.config.dimension = embedding_generator.config.embedding_dim

        # Create storage directory
        ensure_directory(config.vector_db_path)

        # Initialize version_info BEFORE metadata storage is initialized
        self.version_info = {
            "created": time.time(),
            "last_updated": time.time(),
            "num_vectors": 0,
            "version_tag": config.version_tag or f"v{time.strftime('%Y%m%d_%H%M%S')}",
            "embedding_model": embedding_generator.config.model_name if embedding_generator else "unknown"
        }

        # Initialize indices based on configuration
        if config.use_multiple_indices:
            # Create separate indices for each content type
            self.indices = {}
            for content_type in ContentType:
                self.indices[content_type] = self._create_index(content_type.name.lower())

            # Default index for queries or content types without dedicated index
            self.default_index = self._create_index("default")
        else:
            # Single index for all content types
            self.indices = {"default": self._create_index("default")}
            self.default_index = self.indices["default"]

        # Initialize SQL metadata storage if configured
        self.metadata_conn = None
        if config.create_sql_metadata_storage:
            self._init_metadata_storage()

        # Create mapping of chunk IDs to indices for retrieval
        self.chunk_id_to_idx = {}
        self.chunk_id_to_index_name = {}  # Maps chunk_id to index name

        # Store sparse vectors for hybrid search
        self.sparse_vectors = {}

        # Track version information
        self.version_info = {
            "created": time.time(),
            "last_updated": time.time(),
            "num_vectors": 0,
            "version_tag": config.version_tag or f"v{time.strftime('%Y%m%d_%H%M%S')}",
            "embedding_model": embedding_generator.config.model_name if embedding_generator else "unknown"
        }

        # Setup auto-save if configured
        if config.auto_save:
            self._setup_auto_save()

        # Thread lock for index modifications
        self._index_lock = threading.RLock()

        # Batched updates
        self._update_batch = {name: {'vectors': [], 'chunk_ids': []} for name in self.indices}
        self._deletion_batch = {name: set() for name in self.indices}

        # Track deleted items for potential reindexing
        self._deleted_count = 0
        self._total_count = 0

        logger.info(f"Vector store initialized ({config.db_type}, {len(self.indices)} indices)")

    def _create_index(self, name: str) -> Any:
        """Create a vector index based on configuration."""
        # Create a directory for this index
        index_dir = os.path.join(self.config.vector_db_path, name)
        os.makedirs(index_dir, exist_ok=True)

        if self.config.db_type == "faiss":
            return self._create_faiss_index(name)
        elif self.config.db_type == "hnswlib":
            return self._create_hnswlib_index(name)
        elif self.config.custom_index_builder:
            # Use a custom index builder provided in the config
            return self.config.custom_index_builder(self.config, name)
        else:
            raise ValueError(f"Unsupported vector database type: {self.config.db_type}")

    def _create_faiss_index(self, name: str) -> Any:
        """Create a FAISS index."""
        dimension = self.config.dimension

        # Convert metric type to FAISS metric
        if self.config.metric_type == "cosine":
            metric = faiss.METRIC_INNER_PRODUCT
            normalize = True
        elif self.config.metric_type == "l2":
            metric = faiss.METRIC_L2
            normalize = False
        elif self.config.metric_type == "dot":
            metric = faiss.METRIC_INNER_PRODUCT
            normalize = False
        else:
            raise ValueError(f"Unsupported metric type for FAISS: {self.config.metric_type}")

        # Create appropriate index based on type
        if self.config.index_type == "flat":
            # Simple flat index - exhaustive search, most accurate but slowest for large datasets
            if normalize:
                index = faiss.IndexFlatIP(dimension)
            else:
                index = faiss.IndexFlat(dimension, metric)
        elif self.config.index_type == "hnsw":
            # HNSW index - approximate search, good balance of speed and accuracy
            if normalize:
                base_index = faiss.IndexFlatIP(dimension)
            else:
                base_index = faiss.IndexFlat(dimension, metric)

            index = faiss.IndexHNSWFlat(dimension, self.config.m_parameter, metric)
            index.hnsw.efConstruction = self.config.ef_construction
            index.hnsw.efSearch = self.config.ef_search
        elif self.config.index_type == "ivf":
            # IVF index - uses inverted file structure for faster search
            if normalize:
                quantizer = faiss.IndexFlatIP(dimension)
            else:
                quantizer = faiss.IndexFlat(dimension, metric)

            index = faiss.IndexIVFFlat(
                quantizer, dimension, self.config.num_partitions, metric
            )
            # Set the number of probes for search
            index.nprobe = self.config.num_probes
        elif self.config.index_type == "ivfpq":
            # IVF with Product Quantization for memory efficiency
            if normalize:
                quantizer = faiss.IndexFlatIP(dimension)
            else:
                quantizer = faiss.IndexFlat(dimension, metric)

            index = faiss.IndexIVFPQ(
                quantizer, dimension, self.config.num_partitions, 
                self.config.pq_bytes, 8  # 8-bit PQ encoding
            )
            index.nprobe = self.config.num_probes
        else:
            raise ValueError(f"Unsupported FAISS index type: {self.config.index_type}")

        # For IVF-based indices, they need training before use
        if "ivf" in self.config.index_type.lower():
            index.is_trained = False  # Will need training before use

        # Create a wrapper object to hold additional metadata
        index_wrapper = {
            "index": index,
            "name": name,
            "normalize": normalize,
            "metric": metric,
            "idx_to_chunk_id": [],  # Mapping from internal index to chunk IDs
            "trained": index.is_trained if hasattr(index, "is_trained") else True,
            "count": 0  # Number of vectors in the index
        }

        # Load existing index if available
        index_path = os.path.join(self.config.vector_db_path, name, "index.faiss")
        if os.path.exists(index_path):
            try:
                index_wrapper["index"] = faiss.read_index(index_path)

                # Load mapping data
                mapping_path = os.path.join(self.config.vector_db_path, name, "mapping.pkl")
                if os.path.exists(mapping_path):
                    with open(mapping_path, 'rb') as f:
                        index_wrapper["idx_to_chunk_id"] = pickle.load(f)

                index_wrapper["count"] = len(index_wrapper["idx_to_chunk_id"])
                index_wrapper["trained"] = True  # If loaded, assume it's trained

                logger.info(f"Loaded existing FAISS index '{name}' with {index_wrapper['count']} vectors")
            except Exception as e:
                logger.error(f"Error loading FAISS index '{name}': {e}")
                # Keep the new empty index

        return index_wrapper

    def _create_hnswlib_index(self, name: str) -> Any:
        """Create an HNSWLib index."""
        # Create a new HNSW index
        index = hnswlib.Index(space=self.config.metric_type, dim=self.config.dimension)

        # Initialize with parameters
        max_elements = self.config.max_index_size

        # Create a wrapper object
        index_wrapper = {
            "index": index,
            "name": name,
            "max_elements": max_elements,
            "idx_to_chunk_id": [],  # Mapping from internal index to chunk IDs
            "count": 0,  # Number of vectors in the index
            "initialized": False  # Whether index has been initialized
        }

        # Load existing index if available
        index_path = os.path.join(self.config.vector_db_path, name, "index.hnsw")
        if os.path.exists(index_path):
            try:
                index_wrapper["index"].load_index(index_path)

                # Load mapping data
                mapping_path = os.path.join(self.config.vector_db_path, name, "mapping.pkl")
                if os.path.exists(mapping_path):
                    with open(mapping_path, 'rb') as f:
                        index_wrapper["idx_to_chunk_id"] = pickle.load(f)

                index_wrapper["count"] = len(index_wrapper["idx_to_chunk_id"])
                index_wrapper["initialized"] = True

                logger.info(f"Loaded existing HNSW index '{name}' with {index_wrapper['count']} vectors")
            except Exception as e:
                logger.error(f"Error loading HNSW index '{name}': {e}")
                # Initialize a new empty index
                index_wrapper["index"].init_index(
                    max_elements=max_elements,
                    ef_construction=self.config.ef_construction,
                    M=self.config.m_parameter
                )
                index_wrapper["initialized"] = True
        else:
            # Initialize a new empty index
            index_wrapper["index"].init_index(
                max_elements=max_elements,
                ef_construction=self.config.ef_construction,
                M=self.config.m_parameter
            )
            index_wrapper["initialized"] = True

        # Set search parameters
        index_wrapper["index"].set_ef(self.config.ef_search)

        return index_wrapper

    def _init_metadata_storage(self) -> None:
        """Initialize SQLite metadata storage."""
        # Extract the connection string
        if self.config.sql_connection_string.startswith("sqlite:///"):
            db_path = self.config.sql_connection_string[10:]
        else:
            db_path = os.path.join(self.config.vector_db_path, "metadata.db")

        # Create directory if needed
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)

        # Connect to database
        self.metadata_conn = sqlite3.connect(db_path, check_same_thread=False)

        # Create tables if they don't exist
        cursor = self.metadata_conn.cursor()

        # Table for chunk metadata
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id TEXT PRIMARY KEY,
            doc_id TEXT,
            source TEXT,
            section_id TEXT,
            section_title TEXT,
            content_type TEXT,
            content TEXT,
            metadata TEXT
        )
        ''')

        # Table for index metadata
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS index_metadata (
            key TEXT PRIMARY KEY,
            value TEXT
        )
        ''')

        # Create indices for faster queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks (doc_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_chunks_content_type ON chunks (content_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks (source)')

        # Commit changes
        self.metadata_conn.commit()

        # Store initial metadata if table is empty
        cursor.execute('SELECT COUNT(*) FROM index_metadata')
        if cursor.fetchone()[0] == 0:
            self._store_index_metadata('version', json.dumps(self.version_info))
            self._store_index_metadata('config', json.dumps(self.config.__dict__))
            self._store_index_metadata('created_at', str(time.time()))

        # Load version info
        self.version_info = json.loads(self._get_index_metadata('version', '{}'))

        logger.info("Initialized metadata storage")

    def _store_index_metadata(self, key: str, value: str) -> None:
        """Store metadata in SQLite database."""
        if not self.metadata_conn:
            return

        cursor = self.metadata_conn.cursor()
        cursor.execute(
            'INSERT OR REPLACE INTO index_metadata (key, value) VALUES (?, ?)',
            (key, value)
        )
        self.metadata_conn.commit()

    def _get_index_metadata(self, key: str, default: str = None) -> str:
        """Get metadata from SQLite database."""
        if not self.metadata_conn:
            return default

        cursor = self.metadata_conn.cursor()
        cursor.execute('SELECT value FROM index_metadata WHERE key = ?', (key,))
        result = cursor.fetchone()

        return result[0] if result else default

    def _setup_auto_save(self) -> None:
        """Setup automatic saving of indices."""
        def auto_save_loop():
            while True:
                time.sleep(self.config.save_interval)
                try:
                    self.save_indices()
                except Exception as e:
                    logger.error(f"Error during auto-save: {e}")

        # Start auto-save thread
        save_thread = threading.Thread(target=auto_save_loop, daemon=True)
        save_thread.start()

        logger.info(f"Auto-save enabled (interval: {self.config.save_interval}s)")

    def add_chunks(self, chunks: List[Chunk]) -> None:
        """
        Add chunks to the vector store.

        Args:
            chunks: List of chunks with embeddings
        """
        if not chunks:
            return

        # Ensure all chunks have embeddings
        chunks_without_embedding = [chunk for chunk in chunks if chunk.embedding is None]
        if chunks_without_embedding:
            logger.warning(f"{len(chunks_without_embedding)} chunks without embeddings will be skipped")
            chunks = [chunk for chunk in chunks if chunk.embedding is not None]

        if not chunks:
            return

        # Store sparse vectors for hybrid search
        for chunk in chunks:
            if chunk.sparse_embedding:
                self.sparse_vectors[chunk.chunk_id] = chunk.sparse_embedding

        # Group chunks by content type if using multiple indices
        if self.config.use_multiple_indices:
            chunks_by_type = {}
            for chunk in chunks:
                content_type = chunk.content_type
                if content_type not in chunks_by_type:
                    chunks_by_type[content_type] = []
                chunks_by_type[content_type].append(chunk)

            # Add chunks to appropriate indices
            for content_type, type_chunks in chunks_by_type.items():
                if content_type in self.indices:
                    self._add_chunks_to_index(self.indices[content_type], type_chunks)
                else:
                    # Use default index for unknown content types
                    self._add_chunks_to_index(self.default_index, type_chunks)
        else:
            # Add all chunks to the default index
            self._add_chunks_to_index(self.default_index, chunks)

        # Update metadata storage if configured
        if self.config.store_metadata and self.metadata_conn:
            self._store_chunks_metadata(chunks)

        # Update version info
        self.version_info["last_updated"] = time.time()
        self.version_info["num_vectors"] += len(chunks)
        self._total_count += len(chunks)

        if self.config.create_sql_metadata_storage:
            self._store_index_metadata('version', json.dumps(self.version_info))

        logger.info(f"Added {len(chunks)} chunks to vector store")

    def _add_chunks_to_index(self, index_wrapper: Dict[str, Any], chunks: List[Chunk]) -> None:
        """Add chunks to a specific index."""
        with self._index_lock:
            # Handle batched updates if enabled
            if self.config.enable_batched_updates:
                index_name = index_wrapper["name"]

                # Add to the batch
                for chunk in chunks:
                    self._update_batch[index_name]['vectors'].append(chunk.embedding)
                    self._update_batch[index_name]['chunk_ids'].append(chunk.chunk_id)

                    # Track which index contains this chunk
                    self.chunk_id_to_index_name[chunk.chunk_id] = index_name

                # Process batch if it's large enough
                if len(self._update_batch[index_name]['vectors']) >= self.config.batch_size:
                    self._process_update_batch(index_name)

                return

            # For non-batched updates, process immediately
            self._add_vectors_to_index(
                index_wrapper, 
                [chunk.embedding for chunk in chunks],
                [chunk.chunk_id for chunk in chunks]
            )

    def _process_update_batch(self, index_name: str) -> None:
        """Process a batch of updates for an index."""
        with self._index_lock:
            batch = self._update_batch[index_name]
            if not batch['vectors']:
                return

            # Get the index
            index_wrapper = self.indices.get(index_name)
            if not index_wrapper:
                logger.error(f"Index {index_name} not found")
                return

            # Add vectors to index
            self._add_vectors_to_index(
                index_wrapper,
                batch['vectors'],
                batch['chunk_ids']
            )

            # Clear the batch
            self._update_batch[index_name] = {'vectors': [], 'chunk_ids': []}

    def _add_vectors_to_index(self, index_wrapper: Dict[str, Any], vectors: List[List[float]], 
                            chunk_ids: List[str]) -> None:
        """Add vectors to an index."""
        if not vectors:
            return

        index = index_wrapper["index"]
        idx_to_chunk_id = index_wrapper["idx_to_chunk_id"]

        # Convert vectors to numpy array
        vectors_np = np.array(vectors, dtype=np.float32)

        # Handle FAISS indices
        if self.config.db_type == "faiss":
            # Normalize vectors if needed
            if index_wrapper["normalize"]:
                faiss.normalize_L2(vectors_np)

            # Train the index if needed (for IVF indices)
            if hasattr(index, "is_trained") and not index.is_trained and len(vectors_np) >= 10:
                logger.info(f"Training index {index_wrapper['name']} with {len(vectors_np)} vectors")
                index.train(vectors_np)
                index_wrapper["trained"] = True

            # Add vectors
            start_idx = len(idx_to_chunk_id)
            index.add(vectors_np)

            # Update idx_to_chunk_id mapping
            idx_to_chunk_id.extend(chunk_ids)

            # Update chunk_id_to_idx mapping
            for i, chunk_id in enumerate(chunk_ids):
                self.chunk_id_to_idx[chunk_id] = start_idx + i

        # Handle HNSW indices
        elif self.config.db_type == "hnswlib":
            # Add vectors
            start_idx = len(idx_to_chunk_id)

            # Add items by one by one to track the correct indices
            for i, (vector, chunk_id) in enumerate(zip(vectors_np, chunk_ids)):
                index.add_items(vector.reshape(1, -1), [start_idx + i])

            # Update idx_to_chunk_id mapping
            idx_to_chunk_id.extend(chunk_ids)

            # Update chunk_id_to_idx mapping
            for i, chunk_id in enumerate(chunk_ids):
                self.chunk_id_to_idx[chunk_id] = start_idx + i

        # Update count
        index_wrapper["count"] += len(vectors)

    def _store_chunks_metadata(self, chunks: List[Chunk]) -> None:
        """Store chunk metadata in SQLite database."""
        if not self.metadata_conn:
            return

        cursor = self.metadata_conn.cursor()

        for chunk in chunks:
            # Skip if no content or already exists
            if not self.config.store_content:
                chunk_content = ""
            else:
                chunk_content = chunk.content

            # Convert metadata to JSON string
            metadata_json = json.dumps(chunk.metadata) if chunk.metadata else "{}"

            # Store in database
            cursor.execute(
                '''
                INSERT OR REPLACE INTO chunks 
                (chunk_id, doc_id, source, section_id, section_title, content_type, content, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''',
                (
                    chunk.chunk_id,
                    chunk.doc_id,
                    chunk.source,
                    chunk.section_id,
                    chunk.section_title,
                    chunk.content_type.value,
                    chunk_content,
                    metadata_json
                )
            )

        # Commit changes
        self.metadata_conn.commit()

    def search(self, query: Union[str, List[float]], top_k: int = 5, 
              filters: Dict[str, Any] = None, 
              content_type: Optional[ContentType] = None,
              use_hybrid_search: bool = None) -> List[SearchResult]:
        """
        Search for similar chunks.

        Args:
            query: Query string or vector
            top_k: Number of results to return
            filters: Metadata filters for results
            content_type: Optional content type to search within
            use_hybrid_search: Override config setting for hybrid search

        Returns:
            List of search results
        """
        # Process query
        if isinstance(query, str):
            if not self.embedding_generator:
                raise ValueError("Embedding generator required for text queries")

            # Encode query string
            query_vector = self.embedding_generator.encode_query(
                query, content_type=content_type
            )

            # Get sparse vector for hybrid search if needed
            hybrid_search = use_hybrid_search if use_hybrid_search is not None else (
                self.config.hybrid_search_weights and 
                self.config.hybrid_search_weights.get("sparse", 0) > 0 and
                self.sparse_vectors
            )

            if hybrid_search and self.embedding_generator:
                sparse_vector = self.embedding_generator.encode_sparse_query(query)
            else:
                sparse_vector = None
        else:
            # Direct vector query
            query_vector = np.array(query, dtype=np.float32)
            sparse_vector = None
            hybrid_search = False

        # Determine which indices to search
        if content_type and self.config.use_multiple_indices and content_type in self.indices:
            # Search specific content type index
            indices_to_search = [self.indices[content_type]]
        elif self.config.use_multiple_indices:
            # Search all indices
            indices_to_search = list(self.indices.values())
        else:
            # Search default index
            indices_to_search = [self.default_index]

        # Process any batched updates before searching
        if self.config.enable_batched_updates:
            for index_name in [index["name"] for index in indices_to_search]:
                if index_name in self._update_batch and self._update_batch[index_name]['vectors']:
                    self._process_update_batch(index_name)

        # Search each index
        all_results = []

        for index_wrapper in indices_to_search:
            # Skip empty indices
            if index_wrapper["count"] == 0:
                continue

            # Get index-specific results
            index_results = self._search_index(
                index_wrapper, query_vector, top_k, 
                sparse_vector if hybrid_search else None
            )
            all_results.extend(index_results)

        # Sort all results by score
        all_results.sort(key=lambda x: x[1], reverse=True)

        # Apply filters if any
        if filters:
            filtered_results = self._apply_filters(all_results, filters)
        else:
            filtered_results = all_results

        # Get the top results
        top_results = filtered_results[:top_k]

        # Convert to search results with full chunk data
        search_results = [
            self._create_search_result(chunk_id, score, rank)
            for rank, (chunk_id, score) in enumerate(top_results)
        ]

        # Filter out None results (in case some chunks were not found)
        return [result for result in search_results if result is not None]

    def _search_index(self, index_wrapper: Dict[str, Any], query_vector: np.ndarray, 
                    top_k: int, sparse_vector: Optional[Dict[int, float]] = None) -> List[Tuple[str, float]]:
        """Search a specific index."""
        with self._index_lock:
            index = index_wrapper["index"]
            idx_to_chunk_id = index_wrapper["idx_to_chunk_id"]

            # Ensure we don't request more results than available
            actual_top_k = min(top_k, len(idx_to_chunk_id))
            if actual_top_k == 0:
                return []

            # Reshape query vector
            query_vector = query_vector.reshape(1, -1).astype(np.float32)

            # Handle FAISS indices
            if self.config.db_type == "faiss":
                # Normalize query if needed
                if index_wrapper["normalize"]:
                    faiss.normalize_L2(query_vector)

                # Perform search
                D, I = index.search(query_vector, actual_top_k)

                # Convert to chunk_ids and scores
                results = []
                for i in range(len(I[0])):
                    idx = I[0][i]
                    score = float(D[0][i])

                    # Skip invalid indices
                    if idx < 0 or idx >= len(idx_to_chunk_id):
                        continue

                    chunk_id = idx_to_chunk_id[idx]
                    results.append((chunk_id, score))

            # Handle HNSW indices
            elif self.config.db_type == "hnswlib":
                # Perform search
                labels, distances = index.knn_query(query_vector, k=actual_top_k)

                # Convert to chunk_ids and scores
                results = []
                for i in range(len(labels[0])):
                    idx = labels[0][i]

                    # L2 distances need to be converted to similarity scores
                    if self.config.metric_type == "l2":
                        # Convert L2 distance to similarity (smaller distance = higher similarity)
                        score = 1.0 / (1.0 + float(distances[0][i]))
                    else:
                        # For cosine and dot product, higher is better
                        score = float(distances[0][i])

                    # Skip invalid indices
                    if idx < 0 or idx >= len(idx_to_chunk_id):
                        continue

                    chunk_id = idx_to_chunk_id[idx]
                    results.append((chunk_id, score))

            # Apply hybrid search if sparse vector is provided
            if sparse_vector and self.sparse_vectors:
                # Perform sparse vector search
                sparse_results = self._sparse_search(sparse_vector, actual_top_k)

                # Combine dense and sparse results using weighted scores
                vector_weight = self.config.hybrid_search_weights.get("vector", 0.7)
                sparse_weight = self.config.hybrid_search_weights.get("sparse", 0.3)

                # Create dictionaries for easy score lookup
                dense_dict = {chunk_id: score for chunk_id, score in results}
                sparse_dict = {chunk_id: score for chunk_id, score in sparse_results}

                # Combine all chunk_ids
                all_chunk_ids = set(dense_dict.keys()) | set(sparse_dict.keys())

                # Calculate combined scores
                combined_results = []
                for chunk_id in all_chunk_ids:
                    dense_score = dense_dict.get(chunk_id, 0.0)
                    sparse_score = sparse_dict.get(chunk_id, 0.0)

                    # Weighted combination
                    combined_score = (dense_score * vector_weight) + (sparse_score * sparse_weight)
                    combined_results.append((chunk_id, combined_score))

                # Sort by combined score
                results = sorted(combined_results, key=lambda x: x[1], reverse=True)[:actual_top_k]

            return results

    def _sparse_search(self, query_sparse: Dict[int, float], top_k: int) -> List[Tuple[str, float]]:
        """Perform sparse vector search for hybrid retrieval."""
        if not self.sparse_vectors:
            return []

        # Calculate scores against all sparse vectors
        scores = []
        for chunk_id, sparse_vector in self.sparse_vectors.items():
            # Dot product of sparse vectors
            score = sum(query_sparse.get(idx, 0.0) * val for idx, val in sparse_vector.items())
            scores.append((chunk_id, score))

        # Sort by score and return top_k
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def _apply_filters(self, results: List[Tuple[str, float]], 
                     filters: Dict[str, Any]) -> List[Tuple[str, float]]:
        """Apply metadata filters to search results."""
        # If no metadata storage or no filters, return as is
        if not self.metadata_conn or not filters:
            return results

        filtered_results = []

        for chunk_id, score in results:
            # Check if chunk passes all filters
            if self._chunk_passes_filters(chunk_id, filters):
                filtered_results.append((chunk_id, score))

        return filtered_results

    def _chunk_passes_filters(self, chunk_id: str, filters: Dict[str, Any]) -> bool:
        """Check if a chunk passes all filters."""
        if not self.metadata_conn:
            return True

        cursor = self.metadata_conn.cursor()

        # Query for this chunk's metadata
        cursor.execute(
            'SELECT doc_id, source, section_id, section_title, content_type, metadata FROM chunks WHERE chunk_id = ?',
            (chunk_id,)
        )
        result = cursor.fetchone()

        if not result:
            return False

        doc_id, source, section_id, section_title, content_type, metadata_json = result

        # Parse metadata JSON
        metadata = safe_json_loads(metadata_json, {})

        # Check each filter
        for key, value in filters.items():
            # Handle special filter keys
            if key == "doc_id" and doc_id != value:
                return False
            elif key == "source" and source != value:
                return False
            elif key == "section_id" and section_id != value:
                return False
            elif key == "section_title" and section_title != value:
                return False
            elif key == "content_type" and content_type != value:
                return False
            # Handle metadata fields
            elif key in metadata:
                # Support for list/set membership queries
                if isinstance(value, list):
                    if metadata[key] not in value:
                        return False
                # Support for range queries
                elif isinstance(value, dict) and ('min' in value or 'max' in value):
                    if 'min' in value and metadata[key] < value['min']:
                        return False
                    if 'max' in value and metadata[key] > value['max']:
                        return False
                # Simple equality check
                elif metadata[key] != value:
                    return False

        return True

    def _create_search_result(self, chunk_id: str, score: float, rank: int) -> Optional[SearchResult]:
        """Create a SearchResult object for a chunk."""
        # Get chunk data from metadata storage
        chunk = self._get_chunk_by_id(chunk_id)
        if not chunk:
            return None

        # Create search result
        return SearchResult(
            chunk=chunk,
            score=score,
            rank=rank,
            metadata={"retrieval_score": score}
        )

    def _get_chunk_by_id(self, chunk_id: str) -> Optional[Chunk]:
        """Get a chunk by its ID."""
        if not self.metadata_conn:
            # No metadata storage, can't retrieve full chunk
            return None

        cursor = self.metadata_conn.cursor()

        # Query for chunk data
        cursor.execute(
            '''
            SELECT doc_id, source, section_id, section_title, content_type, content, metadata
            FROM chunks WHERE chunk_id = ?
            ''',
            (chunk_id,)
        )
        result = cursor.fetchone()

        if not result:
            return None

        doc_id, source, section_id, section_title, content_type_str, content, metadata_json = result

        # Parse metadata JSON
        metadata = safe_json_loads(metadata_json, {})

        # Convert content type string to enum
        try:
            content_type = ContentType(content_type_str)
        except:
            content_type = ContentType.TEXT

        # Create chunk object
        chunk = Chunk(
            chunk_id=chunk_id,
            content=content,
            doc_id=doc_id,
            source=source,
            section_id=section_id,
            section_title=section_title,
            content_type=content_type,
            metadata=metadata
        )

        return chunk

    def delete_chunks(self, chunk_ids: List[str]) -> int:
        """
        Delete chunks from the vector store.

        Args:
            chunk_ids: List of chunk IDs to delete

        Returns:
            Number of chunks successfully deleted
        """
        if not chunk_ids:
            return 0

        # Handle batched updates if enabled
        if self.config.enable_batched_updates:
            # Add to deletion batch and return
            deleted_count = 0
            for chunk_id in chunk_ids:
                if chunk_id in self.chunk_id_to_index_name:
                    index_name = self.chunk_id_to_index_name[chunk_id]
                    if index_name in self._deletion_batch:
                        self._deletion_batch[index_name].add(chunk_id)
                        deleted_count += 1

            # If batch is large enough, process it
            for index_name, batch in self._deletion_batch.items():
                if len(batch) >= self.config.batch_size:
                    self._process_deletion_batch(index_name)

            return deleted_count

        # Delete from each index
        deleted_count = 0

        # For FAISS, we need to rebuild the index as it doesn't support individual deletion
        # For HNSW, we mark items as deleted
        for index_name, index_wrapper in self.indices.items():
            # Get chunks to delete from this index
            index_chunk_ids = set()
            for chunk_id in chunk_ids:
                if chunk_id in self.chunk_id_to_idx:
                    index_chunk_ids.add(chunk_id)

            if not index_chunk_ids:
                continue

            # Delete from index
            if self.config.db_type == "faiss":
                # FAISS doesn't support deletion directly, need to rebuild
                # For now, mark as deleted in the mapping and rebuild during save or if too many deletions
                for chunk_id in index_chunk_ids:
                    if chunk_id in self.chunk_id_to_idx:
                        del self.chunk_id_to_idx[chunk_id]
                        deleted_count += 1

                # Mark for rebuild if too many deletions
                self._deleted_counter = ThreadSafeCounter()
                self._deleted_counter.increment(len(index_chunk_ids))
                if self._deleted_count / max(1, self._total_count) > self.config.auto_reindex_threshold:
                    logger.info(f"Scheduling index rebuild due to high deletion rate ({self._deleted_count} / {self._total_count})")
                    # This would trigger a rebuild on the next save

            elif self.config.db_type == "hnswlib":
                # HNSW supports markings items as deleted
                try:
                    for chunk_id in index_chunk_ids:
                        if chunk_id in self.chunk_id_to_idx:
                            index = index_wrapper["index"]
                            index.mark_deleted(self.chunk_id_to_idx[chunk_id])
                            del self.chunk_id_to_idx[chunk_id]
                            deleted_count += 1
                except Exception as e:
                    logger.error(f"Error deleting chunks from HNSW index: {e}")

        # Delete from metadata storage
        if self.metadata_conn:
            cursor = self.metadata_conn.cursor()
            for chunk_id in chunk_ids:
                cursor.execute('DELETE FROM chunks WHERE chunk_id = ?', (chunk_id,))

                # Also remove from sparse vectors if present
                if chunk_id in self.sparse_vectors:
                    del self.sparse_vectors[chunk_id]

            self.metadata_conn.commit()

        # Update version info
        self.version_info["last_updated"] = time.time()
        self.version_info["num_vectors"] -= deleted_count

        if self.config.create_sql_metadata_storage:
            self._store_index_metadata('version', json.dumps(self.version_info))

        return deleted_count

    def _process_deletion_batch(self, index_name: str) -> None:
        """Process a batch of deletions for an index."""
        if index_name not in self._deletion_batch or not self._deletion_batch[index_name]:
            return

        chunk_ids = list(self._deletion_batch[index_name])

        # Delete from index
        if self.config.db_type == "faiss":
            # FAISS doesn't support deletion directly, need to rebuild
            # For now, mark as deleted in the mapping
            for chunk_id in chunk_ids:
                if chunk_id in self.chunk_id_to_idx:
                    del self.chunk_id_to_idx[chunk_id]

            # Mark for rebuild
            self._deleted_count += len(chunk_ids)

        elif self.config.db_type == "hnswlib":
            # HNSW supports markings items as deleted
            try:
                index_wrapper = self.indices.get(index_name)
                if index_wrapper:
                    index = index_wrapper["index"]
                    for chunk_id in chunk_ids:
                        if chunk_id in self.chunk_id_to_idx:
                            index.mark_deleted(self.chunk_id_to_idx[chunk_id])
                            del self.chunk_id_to_idx[chunk_id]
            except Exception as e:
                logger.error(f"Error processing deletion batch for HNSW index: {e}")

        # Delete from metadata storage
        if self.metadata_conn:
            cursor = self.metadata_conn.cursor()
            for chunk_id in chunk_ids:
                cursor.execute('DELETE FROM chunks WHERE chunk_id = ?', (chunk_id,))

                # Also remove from sparse vectors if present
                if chunk_id in self.sparse_vectors:
                    del self.sparse_vectors[chunk_id]

            self.metadata_conn.commit()

        # Clear the batch
        self._deletion_batch[index_name] = set()

    def save_indices(self) -> None:
        """Save all indices to disk."""
        with self._index_lock:
            # Process any pending batches
            if self.config.enable_batched_updates:
                for index_name in list(self._update_batch.keys()):
                    if self._update_batch[index_name]['vectors']:
                        self._process_update_batch(index_name)

                for index_name in list(self._deletion_batch.keys()):
                    if self._deletion_batch[index_name]:
                        self._process_deletion_batch(index_name)

            # Check if we need to rebuild any indices
            if self._deleted_count / max(1, self._total_count) > self.config.auto_reindex_threshold:
                logger.info(f"Rebuilding indices due to high deletion rate ({self._deleted_count} / {self._total_count})")
                self._rebuild_indices()
                return

            # Save each index
            for name, index_wrapper in self.indices.items():
                index_dir = os.path.join(self.config.vector_db_path, name)
                os.makedirs(index_dir, exist_ok=True)

                # Backup existing index if configured
                if self.config.backup_before_save:
                    self._backup_index(name)

                # Save index based on type
                if self.config.db_type == "faiss":
                    index_path = os.path.join(index_dir, "index.faiss")
                    try:
                        faiss.write_index(index_wrapper["index"], index_path)

                        # Save mapping
                        mapping_path = os.path.join(index_dir, "mapping.pkl")
                        with open(mapping_path, 'wb') as f:
                            pickle.dump(index_wrapper["idx_to_chunk_id"], f)

                        logger.info(f"Saved FAISS index '{name}' with {index_wrapper['count']} vectors")
                    except Exception as e:
                        logger.error(f"Error saving FAISS index '{name}': {e}")

                elif self.config.db_type == "hnswlib":
                    index_path = os.path.join(index_dir, "index.hnsw")
                    try:
                        index_wrapper["index"].save_index(index_path)

                        # Save mapping
                        mapping_path = os.path.join(index_dir, "mapping.pkl")
                        with open(mapping_path, 'wb') as f:
                            pickle.dump(index_wrapper["idx_to_chunk_id"], f)

                        logger.info(f"Saved HNSW index '{name}' with {index_wrapper['count']} vectors")
                    except Exception as e:
                        logger.error(f"Error saving HNSW index '{name}': {e}")

            # Save sparse vectors if used
            if self.sparse_vectors:
                sparse_path = os.path.join(self.config.vector_db_path, "sparse_vectors.pkl")
                try:
                    with open(sparse_path, 'wb') as f:
                        pickle.dump(self.sparse_vectors, f)
                except Exception as e:
                    logger.error(f"Error saving sparse vectors: {e}")

            # Update version info
            self.version_info["last_updated"] = time.time()
            self.version_info["last_saved"] = time.time()

            if self.config.create_sql_metadata_storage:
                self._store_index_metadata('version', json.dumps(self.version_info))
                self._store_index_metadata('last_saved', str(time.time()))

    def _backup_index(self, name: str) -> None:
        """Backup an index before saving."""
        index_dir = os.path.join(self.config.vector_db_path, name)

        # Skip if index directory doesn't exist
        if not os.path.exists(index_dir):
            return

        # Create backup directory
        backup_dir = os.path.join(self.config.vector_db_path, "backups", name)
        os.makedirs(backup_dir, exist_ok=True)

        # Create timestamped backup
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(backup_dir, f"backup_{timestamp}")

        try:
            # Copy index files
            os.makedirs(backup_path, exist_ok=True)

            for file_name in os.listdir(index_dir):
                src_path = os.path.join(index_dir, file_name)
                dst_path = os.path.join(backup_path, file_name)

                if os.path.isfile(src_path):
                    shutil.copy2(src_path, dst_path)

            logger.info(f"Created backup of index '{name}' at {backup_path}")

            # Remove old backups if needed
            self._prune_backups(name)
        except Exception as e:
            logger.error(f"Error creating backup of index '{name}': {e}")

    def _prune_backups(self, name: str) -> None:
        """Remove old backups to save space."""
        backup_dir = os.path.join(self.config.vector_db_path, "backups", name)

        # Skip if backup directory doesn't exist
        if not os.path.exists(backup_dir):
            return

        # Get list of backups and sort by timestamp (newest first)
        backups = []
        for entry in os.listdir(backup_dir):
            entry_path = os.path.join(backup_dir, entry)
            if os.path.isdir(entry_path) and entry.startswith("backup_"):
                backups.append((entry_path, os.path.getmtime(entry_path)))

        backups.sort(key=lambda x: x[1], reverse=True)

        # Keep only the latest max_backups
        for backup_path, _ in backups[self.config.max_backups:]:
            try:
                shutil.rmtree(backup_path)
                logger.info(f"Removed old backup: {backup_path}")
            except Exception as e:
                logger.error(f"Error removing old backup {backup_path}: {e}")

    def _rebuild_indices(self) -> None:
        """Rebuild indices to reclaim space from deleted items."""
        with self._index_lock:
            # Save sparse vectors
            sparse_vectors_backup = self.sparse_vectors.copy()

            # For each index
            for name, index_wrapper in self.indices.items():
                # Skip empty indices
                if index_wrapper["count"] == 0:
                    continue

                # Create a new index
                new_index_wrapper = self._create_index(f"{name}_new")

                # Get valid chunk IDs and their vectors
                valid_chunk_ids = []
                valid_indices = []

                for i, chunk_id in enumerate(index_wrapper["idx_to_chunk_id"]):
                    if chunk_id in self.chunk_id_to_idx:
                        valid_chunk_ids.append(chunk_id)
                        valid_indices.append(i)

                # Skip if no valid chunks
                if not valid_chunk_ids:
                    continue

                # Get vectors for valid chunks
                if self.config.db_type == "faiss":
                    # FAISS: extract vectors
                    valid_vectors = []
                    for i in valid_indices:
                        vector = faiss.vector_to_array(index_wrapper["index"].reconstruct(i))
                        valid_vectors.append(vector)

                elif self.config.db_type == "hnswlib":
                    # HNSW: we need to reinitialize completely
                    valid_vectors = []
                    for i in valid_indices:
                        # Retrieve vector from the index
                        vector = index_wrapper["index"].get_items([i])[0]
                        valid_vectors.append(vector)

                # Add to new index
                self._add_vectors_to_index(
                    new_index_wrapper,
                    valid_vectors,
                    valid_chunk_ids
                )

                # Replace old index with new one
                self.indices[name] = new_index_wrapper

                # Update chunk_id_to_idx mapping
                for i, chunk_id in enumerate(valid_chunk_ids):
                    self.chunk_id_to_idx[chunk_id] = i

            # Reset deletion counter
            self._deleted_count = 0
            self._total_count = sum(index["count"] for index in self.indices.values())

            # Restore sparse vectors
            self.sparse_vectors = sparse_vectors_backup

            # Save rebuilt indices
            self.save_indices()

            logger.info(f"Rebuilt indices with {self._total_count} vectors")

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        stats = {
            "total_vectors": sum(index["count"] for index in self.indices.values()),
            "indices": {},
            "version_info": self.version_info,
            "db_type": self.config.db_type,
            "index_type": self.config.index_type,
            "sparse_vectors": len(self.sparse_vectors),
            "deleted_count": self._deleted_count,
            "pending_updates": sum(len(batch['vectors']) for batch in self._update_batch.values()),
            "pending_deletions": sum(len(batch) for batch in self._deletion_batch.values())
        }

        # Add index-specific stats
        for name, index_wrapper in self.indices.items():
            stats["indices"][name] = {
                "count": index_wrapper["count"],
                "type": self.config.db_type
            }

            if self.config.db_type == "faiss":
                # Add FAISS-specific stats
                stats["indices"][name]["trained"] = index_wrapper["trained"]

            elif self.config.db_type == "hnswlib":
                # Add HNSW-specific stats
                stats["indices"][name]["ef_search"] = index_wrapper["index"].get_ef()
                stats["indices"][name]["max_elements"] = index_wrapper["max_elements"]

        # Add metadata stats if available
        if self.metadata_conn:
            cursor = self.metadata_conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM chunks')
            stats["metadata_chunks"] = cursor.fetchone()[0]

        return stats

    # def close(self) -> None:
    #     """Close the vector store and release resources."""
    #     # Save indices if configured
    #     if self.config.auto_save:
    #         self.save_indices()

    #     # Close metadata connection
    #     if self.metadata_conn:
    #         self.metadata_conn.close()
    #         self.metadata_conn = None

    #     logger.info("Vector store closed")

    def close(self) -> None:
        """Close the vector store and release resources."""
        # Save indices if configured
        if self.config.auto_save:
            self.save_indices()
            
        # Clear references to FAISS indices
        for name in list(self.indices.keys()):
            if self.config.db_type == "faiss":  # Use self.config.db_type instead of self.db_type
                # Set the reference to None
                self.indices[name]["index"] = None
            
        # Explicitly delete indices
        self.indices.clear()
        
        # Close metadata connection
        if self.metadata_conn:
            self.metadata_conn.close()
            self.metadata_conn = None
            
        logger.info("Vector store closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

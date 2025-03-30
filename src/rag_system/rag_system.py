# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 Anant Patankar

"""
RAG System Core: Where it all comes together

This module integrates all RAG components into a cohesive system:
    - Document processor: Extracts structure from files
    - Chunker: Breaks documents into manageable pieces
    - Embedder: Turns text into vectors
    - Vector store: Indexes and retrieves relevant chunks
    - LLM connector: Generates answers based on retrieved context

Basically, this is the conductor that makes all the instruments play in harmony.
It handles the flow of data through the pipeline and manages all the messy details
like caching, parallel processing, and system state.

Quick start:
> config = RagSystemConfig()  # Create with defaults or customize as needed
> rag = RagSystem(config)     # Fire up the system
> rag.process_document("your_doc.pdf")  # Feed it some knowledge
> answer = rag.query("What's in that document?")  # Get answers

Pro tip: Enable caching. Trust me, your users will thank you when their
follow-up questions come back instantly instead of making them wait.
"""

import os
import json
import logging
import time
import tempfile
import requests
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field, asdict
from pathlib import Path
import threading
import queue
import numpy as np

# Import the important classes
from src.rag_system.document_processor import DocumentProcessor
from src.rag_system.document_chunker import DocumentChunker
from src.rag_system.embedding_generator import EmbeddingGenerator
from src.rag_system.vector_store import VectorStore
from src.utils.helpers import Timer, cosine_similarity, safe_serialize

from src.config.config import RagSystemConfig
from src.config.data_models import QueryResult, Chunk, SearchResult
from src.config.doc_process_config import ProcessedDocument

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('rag_system')



class RagSystem:
    """
    Retrieval-Augmented Generation system.

    This class if for integrating all the RAG system components:
        - Document processing (PDF, HTML, text, etc.)
        - Document chunking (splitting into retrievable pieces)
        - Embedding generation (Creating Vectors from Chunk)
        - Vector storage and retrieval (finding relevant chunks)
        - LLM generation (Generating answers using retrieved context)
    """

    def __init__(self, config: RagSystemConfig):
        """
        Initializes the RAG system.

        Args:
            config: Complete RAG system configuration
        """
        self.config = config

        # Set up logging
        log_level = getattr(logging, config.log_level.upper(), logging.INFO)
        logger.setLevel(log_level)

        # Create cache directory
        os.makedirs(config.cache_dir, exist_ok=True)

        # Initialize components
        logger.info("Initializing RAG system components")

        # Document processor
        self.document_processor = DocumentProcessor(config.processing_config)

        # Document chunker
        self.document_chunker = DocumentChunker(config.chunking_config)

        # Embedding generator
        self.embedding_generator = EmbeddingGenerator(config.embedding_config)
        
        # Initialize sparse vectorizer if hybrid search is enabled
        if config.query_config.use_hybrid_search and hasattr(self.embedding_generator, 'sparse_vectorizer'):
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
                if self.embedding_generator.sparse_vectorizer is None:
                    self.embedding_generator.sparse_vectorizer = TfidfVectorizer(
                        lowercase=True,
                        stop_words='english',
                        ngram_range=(1, 2),
                        max_features=50000
                    )
                    # We'll fit it on first use
                    logger.info("Initialized sparse vectorizer for hybrid search")
            except ImportError:
                logger.warning("scikit-learn not available, hybrid search will not work")
                config.query_config.use_hybrid_search = False

        # Vector store
        self.vector_store = VectorStore(
            config.vector_db_config, 
            embedding_generator=self.embedding_generator
        )

        # Query response cache
        self.query_cache = {}
        if config.enable_caching:
            self._load_query_cache()

        # Worker thread pool for document processing
        self.worker_pool = None
        self.task_queue = None
        self.result_queue = None
        if config.num_threads > 1:
            self._setup_worker_pool()

        # Track system stats
        self.stats = {
            "documents_processed": 0,
            "chunks_created": 0,
            "queries_processed": 0,
            "queries_cached": 0,
            "total_processing_time": 0,
            "total_query_time": 0,
            "last_document_time": 0,
            "last_query_time": 0,
            "system_start_time": time.time()
        }

        logger.info(f"RAG system initialized: {config.system_name} {config.system_version}")

    def _setup_worker_pool(self):
        """Set up worker thread pool for parallel processing."""
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.worker_pool = []

        for _ in range(self.config.num_threads):
            worker = threading.Thread(target=self._worker_loop, daemon=True)
            worker.start()
            self.worker_pool.append(worker)

        logger.info(f"Started worker pool with {self.config.num_threads} threads")

    def _worker_loop(self):
        """Worker thread function for processing documents."""
        while True:
            try:
                task = self.task_queue.get()
                if task is None:  # Shutdown signal
                    break

                task_type, args, kwargs = task

                if task_type == "process_document":
                    file_path = args[0]
                    try:
                        document = self.document_processor.process_document(file_path)
                        chunks = self.document_chunker.create_chunks(document)
                        chunks_with_embeddings = self.embedding_generator.generate_embeddings(chunks)
                        result = (file_path, document, chunks_with_embeddings)
                        self.result_queue.put(("success", result))
                    except Exception as e:
                        self.result_queue.put(("error", (file_path, str(e))))
                else:
                    logger.warning(f"Unknown task type: {task_type}")

                self.task_queue.task_done()
            except Exception as e:
                logger.error(f"Error in worker thread: {e}")

    def _load_query_cache(self):
        """Load query cache from disk."""
        cache_path = os.path.join(self.config.cache_dir, "query_cache.json")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)

                # Convert dictionary to QueryResult objects
                self.query_cache = {
                    key: QueryResult.from_dict(value) 
                    for key, value in cache_data.items()
                }
                
                logger.info(f"Loaded query cache with {len(self.query_cache)} entries")
            except Exception as e:
                logger.warning(f"Failed to load query cache: {e}")
                self.query_cache = {}

    def _save_query_cache(self):
        """Save query cache to disk."""
        if not self.config.enable_caching:
            return

        cache_path = os.path.join(self.config.cache_dir, "query_cache.json")
        try:
            # Convert QueryResult objects to dictionaries
            cache_data = {
                key: safe_serialize(result) 
                for key, result in self.query_cache.items()
            }
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
                
            logger.info(f"Saved query cache with {len(self.query_cache)} entries")
        except Exception as e:
            logger.warning(f"Failed to save query cache: {e}")

    def process_document(self, file_path: str) -> Tuple[ProcessedDocument, List[Chunk]]:
        """
        Process a document file and add it to the RAG system.

        Args:
            file_path: Path to the document file

        Returns:
            Tuple of (processed document, list of chunks with embeddings)
        """
        logger.info(f"Processing document: {file_path}")

        try:
            with Timer("Document processing") as timer:
                # Process document
                document = self.document_processor.process_document(file_path)

                # Create chunks
                chunks = self.document_chunker.create_chunks(document)
                logger.info(f"Created {len(chunks)} chunks from document")

                # Generate embeddings
                chunks_with_embeddings = self.embedding_generator.generate_embeddings(chunks)

                # Add to vector store
                self.vector_store.add_chunks(chunks_with_embeddings)

                # Update stats
                self.stats["documents_processed"] += 1
                self.stats["chunks_created"] += len(chunks)

            processing_time = timer.elapsed
            self.stats["last_document_time"] = processing_time
            self.stats["total_processing_time"] += self.stats["last_document_time"]

            logger.info(f"Document processed successfully in {self.stats['last_document_time']:.2f}s")
            return document, chunks_with_embeddings

        except Exception as e:
            logger.error(f"Error processing document {file_path}: {e}")
            raise

    def process_documents_parallel(self, file_paths: List[str]) -> List[Tuple[str, bool, str]]:
        """
        Process multiple documents in parallel using worker threads.

        Args:
            file_paths: List of paths to document files

        Returns:
            List of tuples (file_path, success, message)
        """
        if not self.worker_pool:
            logger.warning("Worker pool not initialized, processing sequentially")
            results = []
            for file_path in file_paths:
                try:
                    self.process_document(file_path)
                    results.append((file_path, True, "Success"))
                except Exception as e:
                    results.append((file_path, False, str(e)))
            return results

        # Queue tasks
        for file_path in file_paths:
            self.task_queue.put(("process_document", (file_path,), {}))

        # Wait for all tasks to complete
        self.task_queue.join()

        # Collect results
        results = []
        while not self.result_queue.empty():
            status, result = self.result_queue.get()
            if status == "success":
                file_path, document, chunks = result
                # Add to vector store
                self.vector_store.add_chunks(chunks)
                results.append((file_path, True, f"Processed {len(chunks)} chunks"))
            else:
                file_path, error = result
                results.append((file_path, False, error))

        return results

    def query(self, query_text: str, filters: Dict[str, Any] = None) -> QueryResult:
        """
        Execute a RAG query against the system.

        Args:
            query_text: The query text
            filters: Optional filters for retrieval

        Returns:
            QueryResult containing the answer and context information
        """
        timer = Timer("Document processing")
        timer.start()
        logger.info(f"Processing query: {query_text}")

        # Check cache if enabled
        cache_key = f"{query_text}_{json.dumps(filters or {})}"
        if self.config.enable_caching and cache_key in self.query_cache:
            self.stats["queries_cached"] += 1
            cached_result = self.query_cache[cache_key]
            logger.info(f"Query found in cache, returning cached result")
            return cached_result

        try:
            # Build complete filters
            complete_filters = {}
            if filters:
                complete_filters.update(filters)

            # Add filters from query config
            query_config = self.config.query_config
            if query_config.filter_by_metadata:
                complete_filters.update(query_config.filter_by_metadata)

            # Ensure sparse vectorizer is initialized if hybrid search is enabled
            use_hybrid = query_config.use_hybrid_search
            if use_hybrid and hasattr(self.embedding_generator, 'sparse_vectorizer'):
                if (self.embedding_generator.sparse_vectorizer is not None and 
                    not hasattr(self.embedding_generator.sparse_vectorizer, 'vocabulary_')):
                    # Need to fit the vectorizer on some text
                    try:
                        # Get some text from the first 100 chunks in the vector store
                        sample_chunks = []
                        for index_wrapper in self.vector_store.indices.values():
                            if hasattr(index_wrapper, "idx_to_chunk_id"):
                                chunk_ids = index_wrapper.get("idx_to_chunk_id", [])[:100]
                                for chunk_id in chunk_ids:
                                    chunk = self.vector_store._get_chunk_by_id(chunk_id)
                                    if chunk:
                                        sample_chunks.append(chunk.content)
                        
                        if sample_chunks:
                            logger.info(f"Fitting sparse vectorizer on {len(sample_chunks)} sample chunks")
                            self.embedding_generator.sparse_vectorizer.fit(sample_chunks)
                        else:
                            logger.warning("No sample chunks found, disabling hybrid search for this query")
                            use_hybrid = False
                    except Exception as e:
                        logger.warning(f"Error fitting sparse vectorizer: {e}")
                        use_hybrid = False

            # Retrieve relevant chunks
            retrieved_chunks = self.vector_store.search(
                query_text,
                top_k=query_config.max_retrieved_chunks,
                filters=complete_filters,
                use_hybrid_search=use_hybrid
            )

            # Filter by minimum score
            filtered_chunks = [
                result for result in retrieved_chunks
                if result.score >= query_config.min_relevance_score
            ]

            # Apply MMR if enabled to get diverse results
            if query_config.use_mmr and len(filtered_chunks) > 2:
                filtered_chunks = self._apply_mmr(filtered_chunks, query_text)

            # Limit context chunks for generation
            context_chunks = filtered_chunks[:self.config.generation_config.num_context_chunks]

            # Generate answer
            answer, token_usage = self._generate_answer(query_text, context_chunks)

            # Create result
            result = QueryResult(
                query=query_text,
                answer=answer,
                context_chunks=[chunk.chunk for chunk in context_chunks],
                retrieved_chunks=retrieved_chunks,
                execution_time=timer.stop(),
                model_name=self.config.generation_config.model_name,
                token_usage=token_usage,
                metadata={
                    "num_chunks_retrieved": len(retrieved_chunks),
                    "num_chunks_used": len(context_chunks),
                    "filters": complete_filters
                }
            )

            # Update stats
            self.stats["queries_processed"] += 1
            self.stats["last_query_time"] = result.execution_time
            self.stats["total_query_time"] += result.execution_time

            # Cache result if enabled
            if self.config.enable_caching:
                self.query_cache[cache_key] = result
                # Periodically save cache
                if self.stats["queries_processed"] % 10 == 0:
                    self._save_query_cache()

            logger.info(f"Query processed successfully in {result.execution_time:.2f}s")
            return result

        except Exception as e:
            logger.error(f"Error processing query: {e}")

            # Create error result
            result = QueryResult(
                query=query_text,
                answer=f"Error: {str(e)}",
                context_chunks=[],
                retrieved_chunks=[],
                execution_time=timer.stop(),
                model_name=self.config.generation_config.model_name,
                error=str(e)
            )

            return result

    def _apply_mmr(self, search_results: List[SearchResult], query: str) -> List[SearchResult]:
        """
        Apply Maximum Marginal Relevance algorithm to get diverse results.

        I implemented this MMR algorithm to prevent redundant information while
        maintaining query relevance in retrieved content.
        Args:
            search_results: List of search results
            query: The query string

        Returns:
            Filtered list of search results
        """

        if not search_results or len(search_results) <= 2:
            return search_results

        # Get query embedding
        query_embedding = self.embedding_generator.encode_query(query)

        # Get embeddings for chunks
        chunk_embeddings = []
        for result in search_results:
            if result.chunk.embedding is not None:
                chunk_embeddings.append(result.chunk.embedding)
            else:
                # Re-embed if not available
                embedding = self.embedding_generator.encode_query(result.chunk.content)
                chunk_embeddings.append(embedding)

        # Apply MMR selection
        mmr_lambda = self.config.query_config.mmr_lambda
        selected_indices = []
        remaining_indices = list(range(len(search_results)))

        # Always select the first result (highest relevance)
        selected_indices.append(0)
        remaining_indices.remove(0)

        # Select remaining results using MMR
        num_to_select = min(
            self.config.generation_config.num_context_chunks, 
            len(search_results)
        )

        while len(selected_indices) < num_to_select and remaining_indices:
            max_mmr_score = -float('inf')
            max_mmr_idx = -1

            for idx in remaining_indices:
                # Relevance term: similarity to query
                relevance = search_results[idx].score

                # Diversity term: max similarity to any selected document
                max_similarity = 0
                for selected_idx in selected_indices:
                    similarity = self._vector_similarity(
                        chunk_embeddings[idx],
                        chunk_embeddings[selected_idx]
                    )
                    max_similarity = max(max_similarity, similarity)

                # MMR score
                mmr_score = mmr_lambda * relevance - (1 - mmr_lambda) * max_similarity

                if mmr_score > max_mmr_score:
                    max_mmr_score = mmr_score
                    max_mmr_idx = idx

            if max_mmr_idx >= 0:
                selected_indices.append(max_mmr_idx)
                remaining_indices.remove(max_mmr_idx)
            else:
                break

        # Return selected results in order
        selected_indices.sort()  # Preserve original ranking order
        return [search_results[i] for i in selected_indices]

    def _vector_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors."""
        return cosine_similarity(np.array(vec1), np.array(vec2))

    def _generate_answer(self, query: str, context_chunks: List[SearchResult]) -> Tuple[str, Dict[str, int]]:
        """
        Generate an answer using the Ollama API.

        Args:
            query: The query string
            context_chunks: Retrieved context chunks

        Returns:
            Tuple of (answer text, token usage)
        """
        gen_config = self.config.generation_config

        # Format context from chunks
        formatted_context = ""
        for i, result in enumerate(context_chunks):
            chunk = result.chunk
            source_info = f"[Source: {chunk.source}"
            if chunk.section_title:
                source_info += f", Section: {chunk.section_title}"
            source_info += f", Score: {result.score:.2f}]"

            formatted_context += f"--- Context Chunk {i+1} ---\n"
            formatted_context += chunk.content + "\n"
            if gen_config.include_sources:
                formatted_context += source_info + "\n"
            formatted_context += "\n"

        # Format prompt using template
        prompt = gen_config.prompt_template.format(
            context=formatted_context,
            query=query
        )

        # Prepare request to Ollama API
        api_url = f"{gen_config.ollama_base_url}/api/generate"

        request_data = {
            "model": gen_config.model_name,
            "prompt": prompt,
            "system": gen_config.system_prompt,
            "temperature": gen_config.temperature,
            "top_p": gen_config.top_p,
            "top_k": gen_config.top_k,
            "num_predict": gen_config.max_tokens,
            "stream": gen_config.streaming
        }

        try:
            if gen_config.streaming:
                # For streaming, we need to collect response parts
                response_chunks = []

                # Make streaming request
                with requests.post(api_url, json=request_data, stream=True, timeout=gen_config.timeout) as r:
                    r.raise_for_status()

                    for line in r.iter_lines():
                        if line:
                            line_json = json.loads(line)
                            if 'response' in line_json:
                                response_chunks.append(line_json['response'])
            
                # Combine chunks
                response_text = "".join(response_chunks)

                # Approximate token usage (Ollama doesn't always provide this in streaming mode)
                token_usage = {
                    "prompt_tokens": len(prompt.split()),
                    "completion_tokens": len(response_text.split()),
                    "total_tokens": len(prompt.split()) + len(response_text.split())
                }

            else:
                # For non-streaming, make a single request
                response = requests.post(api_url, json=request_data, timeout=gen_config.timeout)
                response.raise_for_status()
                response_json = response.json()

                response_text = response_json.get('response', '')

                # Get token usage if available
                token_usage = {
                    "prompt_tokens": response_json.get('prompt_eval_count', 0),
                    "completion_tokens": response_json.get('eval_count', 0),
                    "total_tokens": response_json.get('prompt_eval_count', 0) + response_json.get('eval_count', 0)
                }

            return response_text, token_usage

        except requests.RequestException as e:
            error_msg = f"Error calling Ollama API: {str(e)}"
            logger.error(error_msg)
            return f"Error: {error_msg}", {"total_tokens": 0}

    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        # Update with latest component stats
        vector_store_stats = self.vector_store.get_stats()

        stats = {
            **self.stats,
            "uptime_seconds": time.time() - self.stats["system_start_time"],
            "vector_store": vector_store_stats,
            "cache_size": len(self.query_cache) if self.config.enable_caching else 0,
            "avg_document_time": (self.stats["total_processing_time"] / max(1, self.stats["documents_processed"])),
            "avg_query_time": (self.stats["total_query_time"] / max(1, self.stats["queries_processed"])),
            "config": {
                "system_name": self.config.system_name,
                "system_version": self.config.system_version,
                "model_name": self.config.generation_config.model_name
            }
        }

        return stats

    def save(self) -> None:
        """Save system state to disk."""
        # Save vector store
        self.vector_store.save_indices()

        # Save query cache
        if self.config.enable_caching:
            self._save_query_cache()

        # Save stats
        stats_path = os.path.join(self.config.cache_dir, "system_stats.json")
        try:
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(self.stats, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save system stats: {e}")

        logger.info("System state saved to disk")

    def close(self) -> None:
        """Close the RAG system and release resources."""
        # Stop worker threads
        if self.worker_pool:
            for _ in range(len(self.worker_pool)):
                self.task_queue.put(None)  # Shutdown signal

            for worker in self.worker_pool:
                worker.join(timeout=1.0)

        # Save system state
        self.save()

        # Close vector store
        self.vector_store.close()

        logger.info("RAG system closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

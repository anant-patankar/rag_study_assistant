# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 Anant Patankar

"""
Helper Functions for the RAG system.

This module contains shared utility functions used across different
RAG system components.
"""

import os
import re
import json
import logging
import time
import hashlib
import threading
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import numpy as np

# Configure module-level logger
logger = logging.getLogger('rag_system.utils')

# ===== FILE HANDLING UTILITIES =====

def ensure_directory(path: str) -> str:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path

    Returns:
        The same path (for method chaining)
    """
    os.makedirs(path, exist_ok=True)
    return path

def get_file_extension(file_path: str) -> str:
    """
    Get the lowercase extension of a file.

    Args:
        file_path: Path to the file

    Returns:
        Lowercase extension with dot (e.g., '.pdf')
    """
    return os.path.splitext(file_path)[1].lower()

# ===== TEXT PROCESSING UTILITIES =====

def clean_text(text: str) -> str:
    """
    Clean text by removing excessive whitespace and normalizing line breaks.

    Args:
        text: Input text

    Returns:
        Cleaned text
    """
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Normalize line breaks
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def extract_text_between(text: str, start_marker: str, end_marker: str) -> str:
    """
    Extract text between two markers.

    Args:
        text: Input text
        start_marker: Starting marker
        end_marker: Ending marker

    Returns:
        Extracted text or empty string if markers not found
    """
    pattern = f"{re.escape(start_marker)}(.*?){re.escape(end_marker)}"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1) if match else ""

# ===== VECTOR OPERATIONS =====

def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """
    Normalize a vector to unit length.

    Args:
        vector: Input vector (1-dimensional)

    Returns:
        Normalized vector
    """
    norm = np.linalg.norm(vector)
    if norm > 0:
        return vector / norm
    return vector

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Cosine similarity score (between -1 and 1)
    """
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def batch_items(items: List[Any], batch_size: int) -> List[List[Any]]:
    """
    Split a list into batches of specified size.

    Args:
        items: List of items to batch
        batch_size: Size of each batch

    Returns:
        List of batches
    """
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]

# ===== HASHING & ID GENERATION =====

def generate_hash(content: str) -> str:
    """
    Create deterministic MD5 hash from string content.

    Generates unique identifiers for content-based
    deduplication and reference tracking.

    Args:
        content: Input string

    Returns:
        MD5 hash as hexadecimal string
    """
    return hashlib.md5(content.encode()).hexdigest()

def generate_id(prefix: str = "", content: str = None) -> str:
    """
    Generate a unique ID with optional prefix and content hash.

    Args:
        prefix: Optional prefix for the ID
        content: Optional content to hash

    Returns:
        Unique ID
    """
    if content:
        hash_part = generate_hash(content)[:12]
    else:
        hash_part = hashlib.md5(str(time.time()).encode()).hexdigest()[:12]

    if prefix:
        return f"{prefix}_{hash_part}"
    return hash_part

# ===== CONCURRENCY UTILITIES =====

class ThreadSafeCounter:
    """Thread-safe counter for tracking operations."""

    def __init__(self, initial: int = 0):
        """Initialize counter with optional initial value."""
        self.value = initial
        self._lock = threading.Lock()

    def increment(self, amount: int = 1) -> int:
        """Increment counter and return new value."""
        with self._lock:
            self.value += amount
            return self.value

    def decrement(self, amount: int = 1) -> int:
        """Decrement counter and return new value."""
        with self._lock:
            self.value -= amount
            return self.value

    def get(self) -> int:
        """Get current value."""
        with self._lock:
            return self.value

# ===== JSON UTILITIES =====

def safe_json_loads(json_str: str, default: Any = None) -> Any:
    """
    Safely load JSON string.

    Args:
        json_str: JSON string to parse
        default: Default value if parsing fails

    Returns:
        Parsed JSON or default value
    """
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return default if default is not None else {}

def safe_serialize(obj: Any) -> Dict[str, Any]:
    """
    Convert an object to a serializable dictionary, handling common edge cases.

    Args:
        obj: Object to serialize

    Returns:
        Serializable dictionary
    """
    if hasattr(obj, 'to_dict') and callable(obj.to_dict):
        return obj.to_dict()
    elif hasattr(obj, '__dict__'):
        result = {}
        for key, value in obj.__dict__.items():
            if not key.startswith('_'):  # Skip private attributes
                if isinstance(value, np.ndarray):
                    result[key] = value.tolist()
                elif isinstance(value, (list, tuple)):
                    result[key] = [safe_serialize(item) for item in value]
                elif hasattr(value, 'to_dict'):
                    result[key] = value.to_dict()
                else:
                    result[key] = value
        return result
    return obj

# ===== TIMING AND PROFILING =====

class Timer:
    """Timer for measuring execution time."""
    
    def __init__(self, name: str = ""):
        """Initialize timer with optional name."""
        self.name = name
        self.start_time = None
        self.elapsed = 0
        
    def __enter__(self):
        """Start timer when entering context."""
        self.start_time = time.time()
        return self
        
    def __exit__(self, *args):
        """Stop timer when exiting context."""
        self.elapsed = time.time() - self.start_time
        if self.name:
            logger.debug(f"{self.name} completed in {self.elapsed:.4f}s")
            
    def start(self):
        """Start the timer explicitly."""
        self.start_time = time.time()
        return self
    
    def stop(self):
        """
        Stop the timer and return elapsed time.
        """
        if self.start_time is None:
            logger.warning(f"Timer '{self.name}' stopped without being started")
            return 0
            
        self.elapsed = time.time() - self.start_time
        
        if self.name:
            logger.debug(f"{self.name} completed in {self.elapsed:.4f}s")
            
        return self.elapsed

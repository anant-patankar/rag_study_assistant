# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 Anant Patankar


"""
This is a comprehensive chunking module for RAG systems

This module splits processed documents into optimized chunks for vector database indexing,
addressing common chunking issues in RAG systems.
"""

import logging
import re
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum

import sys
import os

from src.config.doc_process_config import ProcessedDocument, Section
from src.config.data_models import ContentType, Chunk

from nltk.tokenize import sent_tokenize

# Logging configuration
logger = logging.getLogger('rag_system.chunker')

from src.config.config import ChunkingConfig


class DocumentChunker:
    """Class for creating chunks from processed documents."""

    def __init__(self, config: ChunkingConfig):
        """
        Initialization of the document chunker.

        Args:
            config: Chunking configuration
        """
        self.config = config

        # chunking strategies
        self.strategies = {
            'fixed': self._fixed_size_chunking,
            'semantic': self._semantic_chunking,
            'hybrid': self._hybrid_chunking
        }

        # add custom strategies
        for strategy_name, strategy_func in config.custom_chunking_strategies.items():
            self.strategies[strategy_name] = strategy_func

    def create_chunks(self, document: ProcessedDocument) -> List[Chunk]:
        """
        Function used for creating chunks from Processed Document.

        Args:
            document: Processed document

        Returns:
            List of chunks ready for embedding
        """
        logger.info(f"Creating chunks using {self.config.strategy} strategy")

        # Select appropriate chunking strategy
        chunking_func = self.strategies.get(self.config.strategy)
        if not chunking_func:
            raise ValueError(f"Unknown chunking strategy: {self.config.strategy}. "
                        f"Available: {list(self.strategies.keys())}")

        # primary text chunks
        text_chunks = chunking_func(document)

        # Add special content chunks if configured
        chunks = list(text_chunks)

        if self.config.include_special_chunks:
            # table chunks
            if document.tables and self.config.include_table_summaries:
                chunks.extend(self._create_table_chunks(document))

            # image chunks
            if document.images and self.config.include_image_captions:
                chunks.extend(self._create_image_chunks(document))

            # formula chunks
            if document.formulas:
                chunks.extend(self._create_formula_chunks(document))

            # code chunks
            if document.code_blocks:
                chunks.extend(self._create_code_chunks(document))

        # create maximum chunks if configured
        if self.config.max_chunks_per_doc and len(chunks) > self.config.max_chunks_per_doc:
            logger.warning(f"Limiting document to {self.config.max_chunks_per_doc} chunks "
                        f"(from {len(chunks)} original)")
            chunks = chunks[:self.config.max_chunks_per_doc]

        logger.info(f"Created {len(chunks)} chunks total")
        return chunks

    def _fixed_size_chunking(self, document: ProcessedDocument) -> List[Chunk]:
        """Create fixed-size chunks from document."""
        chunks = []

        for section in document.content:
            text = section.content

            # empty sections skipping
            if not text.strip():
                continue

            # create chunks with overlap
            start = 0
            while start < len(text):
                end = start + self.config.chunk_size

                # Respect boundaries if configured
                if end < len(text):
                    if self.config.respect_paragraph_boundaries:
                        # Try to find paragraph boundary
                        para_end = text.find('\n\n', start, end)
                        if para_end != -1:
                            end = para_end + 2  # Include the newlines

                    if self.config.respect_sentence_boundaries:
                        # If no paragraph boundary or still exceeds, try sentence boundary
                        if end >= len(text) or (self.config.respect_paragraph_boundaries and para_end == -1):
                            # Find the last sentence boundary
                            sentence_end = max(
                                text.rfind('. ', start, end),
                                text.rfind('? ', start, end),
                                text.rfind('! ', start, end),
                                text.rfind('.\n', start, end),
                                text.rfind('?\n', start, end),
                                text.rfind('!\n', start, end)
                            )
                            if sentence_end != -1:
                                end = sentence_end + 2  # Include the period and space

                # Don't create tiny chunks at the end
                if end - start < self.config.chunk_size * 0.5 and start > 0:
                    break

                # Ensure we don't exceed text length
                end = min(end, len(text))

                # Get the chunk text
                chunk_text = text[start:end]

                # Skip empty chunks
                if not chunk_text.strip():
                    start = end
                    continue

                # metadata creation
                metadata = {}
                if self.config.include_metadata:
                    metadata.update(section.metadata)
                    metadata['section_level'] = section.level
                    if document.metadata.get('title'):
                        metadata['document_title'] = document.metadata['title']

                # Add breadcrumbs if configured
                if self.config.include_breadcrumbs:
                    breadcrumbs = " > ".join(section.path)
                    chunk_text = f"{breadcrumbs}\n\n{chunk_text}"

                # Add section path if configured
                if self.config.include_section_path:
                    metadata['section_path'] = section.path

                # Create the chunk
                chunk = Chunk(
                    content=chunk_text,
                    doc_id=document.doc_id,
                    source=document.source,
                    section_id=section.section_id,
                    section_title=section.title,
                    content_type=ContentType.TEXT,
                    strategy="fixed",
                    start_pos=start,
                    end_pos=end,
                    metadata=metadata
                )
                chunks.append(chunk)

                # Move to next position with overlap
                start = end - self.config.chunk_overlap

                # added break to avoid infinite loop
                if start >= end - 1:
                    break

        return chunks

    def _semantic_chunking(self, document: ProcessedDocument) -> List[Chunk]:
        """
            This function creates chunks respecting semantic boundaries 
            like sentences and paragraphs.
        """
        chunks = []

        for section in document.content:
            text = section.content

            # Skip empty sections
            if not text.strip():
                continue

            # firstly split into paragraphs
            paragraphs = text.split('\n\n')

            current_chunk = []
            current_size = 0

            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue

                para_len = len(para)

                # If this paragraph alone exceeds chunk_size, we need to split it
                if para_len > self.config.chunk_size:
                    # First add current chunk if not empty
                    if current_chunk:
                        self._add_semantic_chunk(current_chunk, section, document, chunks)
                        current_chunk = []
                        current_size = 0

                    # Now split the paragraph by sentences
                    try:
                        sentences = sent_tokenize(para)
                    except:
                        # Fallback to simple splitting
                        sentences = para.split('. ')
                        sentences = [s + '.' for s in sentences[:-1]] + [sentences[-1]]

                    para_chunk = []
                    para_size = 0

                    for sentence in sentences:
                        sentence = sentence.strip()
                        if not sentence:
                            continue

                        sentence_len = len(sentence)

                        # If single sentence exceeds chunk size, split by words
                        if sentence_len > self.config.chunk_size:
                            # Add current paragraph chunk first
                            if para_chunk:
                                self._add_semantic_chunk(para_chunk, section, document, chunks)
                                para_chunk = []
                                para_size = 0

                            # Split the long sentence by words
                            words = sentence.split()
                            word_chunk = []
                            word_size = 0

                            for word in words:
                                word_len = len(word) + 1  # +1 for space

                                if word_size + word_len <= self.config.chunk_size:
                                    word_chunk.append(word)
                                    word_size += word_len
                                else:
                                    # Add word chunk and start new one
                                    if word_chunk:
                                        chunk_text = ' '.join(word_chunk)
                                        chunk = Chunk(
                                            content=chunk_text,
                                            doc_id=document.doc_id,
                                            source=document.source,
                                            section_id=section.section_id,
                                            section_title=section.title,
                                            content_type=ContentType.TEXT,
                                            strategy="semantic-word",
                                            metadata={
                                                'section_level': section.level,
                                                'section_path': section.path,
                                                'long_sentence': True
                                            }
                                        )
                                        chunks.append(chunk)

                                    word_chunk = [word]
                                    word_size = word_len

                            # Add the last word chunk
                            if word_chunk:
                                chunk_text = ' '.join(word_chunk)
                                chunk = Chunk(
                                    content=chunk_text,
                                    doc_id=document.doc_id,
                                    source=document.source,
                                    section_id=section.section_id,
                                    section_title=section.title,
                                    content_type=ContentType.TEXT,
                                    strategy="semantic-word",
                                    metadata={
                                        'section_level': section.level,
                                        'section_path': section.path,
                                        'long_sentence': True
                                    }
                                )
                                chunks.append(chunk)

                        # Normal case - add sentence to current paragraph chunk
                        elif para_size + sentence_len <= self.config.chunk_size:
                            para_chunk.append(sentence)
                            para_size += sentence_len + 1  # +1 for space
                        else:
                            # Create chunk from current sentences
                            if para_chunk:
                                self._add_semantic_chunk(para_chunk, section, document, chunks)

                            # Start new paragraph chunk
                            para_chunk = [sentence]
                            para_size = sentence_len

                    # Add the last paragraph chunk
                    if para_chunk:
                        self._add_semantic_chunk(para_chunk, section, document, chunks)

                # Normal case - add paragraph to current chunk        
                elif current_size + para_len <= self.config.chunk_size:
                    current_chunk.append(para)
                    current_size += para_len + 2  # +2 for paragraph break
                else:
                    # Current chunk is full, finalize it
                    if current_chunk:
                        self._add_semantic_chunk(current_chunk, section, document, chunks)

                    # Start new chunk with this paragraph
                    current_chunk = [para]
                    current_size = para_len

            # Add the last chunk
            if current_chunk:
                self._add_semantic_chunk(current_chunk, section, document, chunks)

        return chunks

    def _add_semantic_chunk(self, texts: List[str], section: Section, document: ProcessedDocument, 
                            chunks: List[Chunk]) -> None:
        """
            This is helper function to add a semantic chunk from a list of text elements.
        """
        # Join the texts
        if self.config.respect_paragraph_boundaries:
            chunk_text = '\n\n'.join(texts)  # Keep paragraph structure
        else:
            chunk_text = ' '.join(texts)

        # Skip empty chunks
        if not chunk_text.strip():
            return

        # Create metadata
        metadata = {}
        if self.config.include_metadata:
            metadata.update(section.metadata)
            metadata['section_level'] = section.level
            if document.metadata.get('title'):
                metadata['document_title'] = document.metadata['title']

        # Add breadcrumbs if configured
        if self.config.include_breadcrumbs:
            breadcrumbs = " > ".join(section.path)
            chunk_text = f"{breadcrumbs}\n\n{chunk_text}"

        # Add section path if configured
        if self.config.include_section_path:
            metadata['section_path'] = section.path

        # Create the chunk
        chunk = Chunk(
            content=chunk_text,
            doc_id=document.doc_id,
            source=document.source,
            section_id=section.section_id,
            section_title=section.title,
            content_type=ContentType.TEXT,
            strategy="semantic",
            metadata=metadata
        )
        chunks.append(chunk)

    def _hybrid_chunking(self, document: ProcessedDocument) -> List[Chunk]:
        """
        Hybrid chunking strategy that respects section boundaries and tries to keep
        semantically related content together.
        """
        chunks = []

        # If sections are small enough, keep them as single chunks
        if self.config.respect_section_boundaries:
            for section in document.content:
                if len(section.content) <= self.config.chunk_size:
                    # Create metadata
                    metadata = {}
                    if self.config.include_metadata:
                        metadata.update(section.metadata)
                        metadata['section_level'] = section.level
                        if document.metadata.get('title'):
                            metadata['document_title'] = document.metadata['title']

                    # Add section path if configured
                    if self.config.include_section_path:
                        metadata['section_path'] = section.path

                    # Skip if empty
                    if not section.content.strip():
                        continue

                    # Create whole-section chunk
                    chunk_text = section.content

                    # Add breadcrumbs if configured
                    if self.config.include_breadcrumbs:
                        breadcrumbs = " > ".join(section.path)
                        chunk_text = f"{breadcrumbs}\n\n{chunk_text}"

                    chunk = Chunk(
                        content=chunk_text,
                        doc_id=document.doc_id,
                        source=document.source,
                        section_id=section.section_id,
                        section_title=section.title,
                        content_type=ContentType.TEXT,
                        strategy="hybrid-section",
                        metadata=metadata
                    )
                    chunks.append(chunk)

                # For larger sections, use semantic chunking
                else:
                    # Process section with semantic chunking
                    section_chunks = self._semantic_chunking_for_section(section, document)
                    chunks.extend(section_chunks)

        # If not respecting section boundaries, use semantic chunking for the entire document
        else:
            chunks = self._semantic_chunking(document)

        return chunks

    def _semantic_chunking_for_section(self, section: Section, document: ProcessedDocument) -> List[Chunk]:
        """
            This function is created to apply semantic chunking to a single section.
        """
        # Create a temporary mini-document with just this section
        mini_doc = ProcessedDocument(
            source=document.source,
            doc_id=document.doc_id,
            content=[section],
            metadata=document.metadata
        )

        # Apply semantic chunking to this mini-document
        return self._semantic_chunking(mini_doc)

    def _create_table_chunks(self, document: ProcessedDocument) -> List[Chunk]:
        """Create chunks from tables."""
        chunks = []

        for table in document.tables:
            # Get section info
            section_id = table.section_id
            section_title = ""
            section_path = []
            section_level = 1

            for section in document.content:
                if section.section_id == section_id:
                    section_title = section.title
                    section_path = section.path
                    section_level = section.level
                    break

            # Create string representation of the table
            if table.headers:
                table_str = " | ".join(map(str, table.headers)) + "\n"
                table_str += "-" * len(table_str) + "\n"
            else:
                table_str = ""

            for row in table.content:
                table_str += " | ".join(str(cell) for cell in row) + "\n"

            # Add caption and summary if available
            chunk_text = ""
            if table.caption:
                chunk_text += f"Table: {table.caption}\n\n"
            elif table.summary:
                chunk_text += f"{table.summary}\n\n"

            chunk_text += table_str

            # Add breadcrumbs if configured
            if self.config.include_breadcrumbs and section_path:
                breadcrumbs = " > ".join(section_path)
                chunk_text = f"{breadcrumbs}\n\n{chunk_text}"

            # Create chunk for the table
            chunk = Chunk(
                content=chunk_text,
                doc_id=document.doc_id,
                source=document.source,
                section_id=section_id,
                section_title=section_title,
                content_type=ContentType.TABLE,
                special_content_id=table.table_id,
                strategy="table",
                metadata={
                    'document_title': document.metadata.get('title', ''),
                    'section_level': section_level,
                    'section_path': section_path,
                    'content_type': 'table',
                    'has_headers': bool(table.headers),
                    'row_count': len(table.content),
                    'column_count': len(table.headers) if table.headers else (len(table.content[0]) if table.content else 0),
                    'table_id': table.table_id
                }
            )
            chunks.append(chunk)

        return chunks

    def _create_image_chunks(self, document: ProcessedDocument) -> List[Chunk]:
        """
            This function creates chunks from images.
        """
        chunks = []

        for image in document.images:
            # Get section info
            section_id = image.section_id
            section_title = ""
            section_path = []
            section_level = 1

            for section in document.content:
                if section.section_id == section_id:
                    section_title = section.title
                    section_path = section.path
                    section_level = section.level
                    break

            # Create text description of the image
            chunk_text = f"Image: {image.caption if image.caption else 'No caption'}"
            if image.alt_text:
                chunk_text += f"\nAlt text: {image.alt_text}"
            if image.width and image.height:
                chunk_text += f"\nDimensions: {image.width}x{image.height}"
            if image.ocr_text:
                chunk_text += f"\nText content: {image.ocr_text}"

            # Add breadcrumbs if configured
            if self.config.include_breadcrumbs and section_path:
                breadcrumbs = " > ".join(section_path)
                chunk_text = f"{breadcrumbs}\n\n{chunk_text}"

            # Create chunk for the image
            chunk = Chunk(
                content=chunk_text,
                doc_id=document.doc_id,
                source=document.source,
                section_id=section_id,
                section_title=section_title,
                content_type=ContentType.IMAGE,
                special_content_id=image.image_id,
                strategy="image",
                metadata={
                    'document_title': document.metadata.get('title', ''),
                    'section_level': section_level,
                    'section_path': section_path,
                    'content_type': 'image',
                    'image_id': image.image_id,
                    'image_extension': image.extension,
                    'has_ocr': bool(image.ocr_text),
                    'has_caption': bool(image.caption),
                    'has_alt_text': bool(image.alt_text)
                }
            )
            chunks.append(chunk)

        return chunks

    def _create_formula_chunks(self, document: ProcessedDocument) -> List[Chunk]:
        """Create chunks from formulas."""
        chunks = []

        for formula in document.formulas:
            # Get section info
            section_id = formula.section_id
            section_title = ""
            section_path = []
            section_level = 1

            for section in document.content:
                if section.section_id == section_id:
                    section_title = section.title
                    section_path = section.path
                    section_level = section.level
                    break

            # Create text representation of the formula
            formula_type = "Inline formula" if formula.is_inline else "Block formula"
            chunk_text = f"{formula_type}:\n{formula.content}"

            if formula.rendered_text:
                chunk_text += f"\n\nRendered as: {formula.rendered_text}"

            # Add breadcrumbs if configured
            if self.config.include_breadcrumbs and section_path:
                breadcrumbs = " > ".join(section_path)
                chunk_text = f"{breadcrumbs}\n\n{chunk_text}"

            # Create chunk for the formula
            chunk = Chunk(
                content=chunk_text,
                doc_id=document.doc_id,
                source=document.source,
                section_id=section_id,
                section_title=section_title,
                content_type=ContentType.FORMULA,
                special_content_id=formula.formula_id,
                strategy="formula",
                metadata={
                    'document_title': document.metadata.get('title', ''),
                    'section_level': section_level,
                    'section_path': section_path,
                    'content_type': 'formula',
                    'formula_id': formula.formula_id,
                    'is_inline': formula.is_inline,
                    'has_rendered_text': bool(formula.rendered_text)
                }
            )
            chunks.append(chunk)

        return chunks

    def _create_code_chunks(self, document: ProcessedDocument) -> List[Chunk]:
        """Create chunks from code blocks."""
        chunks = []

        for code in document.code_blocks:
            # Get section info
            section_id = code.section_id
            section_title = ""
            section_path = []
            section_level = 1

            for section in document.content:
                if section.section_id == section_id:
                    section_title = section.title
                    section_path = section.path
                    section_level = section.level
                    break

            # Create text representation of the code
            if code.language:
                chunk_text = f"Code ({code.language}):"
            else:
                chunk_text = "Code:"

            if code.caption:
                chunk_text += f" {code.caption}"

            chunk_text += f"\n\n{code.content}"

            # Add breadcrumbs if configured
            if self.config.include_breadcrumbs and section_path:
                breadcrumbs = " > ".join(section_path)
                chunk_text = f"{breadcrumbs}\n\n{chunk_text}"

            # Create chunk for the code
            chunk = Chunk(
                content=chunk_text,
                doc_id=document.doc_id,
                source=document.source,
                section_id=section_id,
                section_title=section_title,
                content_type=ContentType.CODE,
                special_content_id=code.code_id,
                strategy="code",
                metadata={
                    'document_title': document.metadata.get('title', ''),
                    'section_level': section_level,
                    'section_path': section_path,
                    'content_type': 'code',
                    'code_id': code.code_id,
                    'language': code.language,
                    'has_caption': bool(code.caption)
                }
            )
            chunks.append(chunk)

        return chunks

    def save_chunks_jsonl(self, chunks: List[Chunk], output_path: str) -> None:
        """Save chunks to a JSONL file for easy loading into vector databases."""
        import json

        with open(output_path, 'w', encoding='utf-8') as f:
            for chunk in chunks:
                # Don't include embedding in the saved file
                chunk_dict = chunk.to_dict()
                if 'embedding' in chunk_dict:
                    del chunk_dict['embedding']
                if 'sparse_embedding' in chunk_dict:
                    del chunk_dict['sparse_embedding']

                f.write(json.dumps(chunk_dict, ensure_ascii=False) + '\n')

        logger.info(f"Saved {len(chunks)} chunks to {output_path}")

    def get_unique_chunk_metadata_keys(self, chunks: List[Chunk]) -> List[str]:
        """Get a list of all unique metadata keys present in the chunks."""
        keys = set()
        for chunk in chunks:
            keys.update(chunk.metadata.keys())
        return sorted(list(keys))


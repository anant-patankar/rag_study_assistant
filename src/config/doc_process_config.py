# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 Anant Patankar

"""
The RAG system's fundamental data structures used throughout the RAG system for 
representing processed document content and its components. These models serve as
the fundamental building blocks that flow through the RAG pipeline.
"""


from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field, asdict
import json
import pandas as pd
import logging
import io
from PIL import Image
import os
from src.utils.helpers import generate_hash, ensure_directory

logger = logging.getLogger('rag_system')

@dataclass
class TableData:
    """Structured representation of a table."""
    content: List[List[str]]  # Table as a list of rows, each row is a list of cells
    headers: List[str] = field(default_factory=list)
    page_num: Optional[int] = None
    position: Optional[int] = None
    section_id: Optional[str] = None
    table_id: str = ""
    caption: str = ""
    summary: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.table_id:
            # Generate a unique ID based on content
            content_str = json.dumps(self.content)
            self.table_id = generate_hash(content_str)

        # Generate a simple summary if none is provided
        if not self.summary and self.content:
            try:
                num_rows = len(self.content)
                num_cols = len(self.headers) if self.headers else (len(self.content[0]) if self.content else 0)
                if not self.caption:
                    self.summary = f"Table with {num_rows} rows and {num_cols} columns"
                else:
                    self.summary = f"{self.caption} - Table with {num_rows} rows and {num_cols} columns"
            except:
                self.summary = "Table data"

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        if self.headers and len(self.headers) == len(self.content[0]):
            return pd.DataFrame(self.content, columns=self.headers)
        else:
            return pd.DataFrame(self.content)

    def to_markdown(self) -> str:
        """Convert to markdown table format."""
        if not self.content:
            return ""

        result = []

        # Headers
        if self.headers:
            result.append("| " + " | ".join(self.headers) + " |")
            result.append("| " + " | ".join(["---"] * len(self.headers)) + " |")

        # Content
        for row in self.content:
            result.append("| " + " | ".join([str(cell) for cell in row]) + " |")

        return "\n".join(result)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'table_id': self.table_id,
            'content': self.content,
            'headers': self.headers,
            'page_num': self.page_num,
            'position': self.position,
            'section_id': self.section_id,
            'caption': self.caption,
            'summary': self.summary,
            'metadata': self.metadata,
            # Add serialized DataFrame as string for convenience
            'dataframe_repr': self.to_dataframe().to_string(index=False) if self.content else ""
        }

@dataclass
class ImageData:
    """Structured representation of an image."""
    data: bytes  # Binary image data
    extension: str  # File extension (jpg, png, etc.)
    page_num: Optional[int] = None
    image_id: str = ""
    caption: str = ""
    alt_text: str = ""
    width: Optional[int] = None
    height: Optional[int] = None
    section_id: Optional[str] = None
    ocr_text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.image_id:
            # Generate a unique ID based on image data
            self.image_id = generate_hash(self.data)

        # Extract width and height if not provided
        if self.data and (self.width is None or self.height is None):
            try:
                img = Image.open(io.BytesIO(self.data))
                self.width, self.height = img.size
            except Exception as e:
                logger.warning(f"Could not determine image dimensions: {e}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'image_id': self.image_id,
            'extension': self.extension,
            'page_num': self.page_num,
            'caption': self.caption,
            'alt_text': self.alt_text,
            'width': self.width,
            'height': self.height,
            'section_id': self.section_id,
            'ocr_text': self.ocr_text,
            'metadata': self.metadata,
            # Don't include binary data in the dict representation
            'data_size': len(self.data) if self.data else 0
        }

    def get_description(self) -> str:
        """Get a textual description of the image."""
        desc = f"Image: {self.caption if self.caption else 'No caption'}"
        if self.alt_text:
            desc += f" | Alt text: {self.alt_text}"
        if self.width and self.height:
            desc += f" | Dimensions: {self.width}x{self.height}"
        if self.ocr_text:
            desc += f" | OCR text: {self.ocr_text[:100]}" + ("..." if len(self.ocr_text) > 100 else "")
        return desc

@dataclass
class FormulaData:
    """Structured representation of a mathematical formula."""
    content: str  # LaTeX or MathML content
    formula_id: str = ""
    is_inline: bool = True
    page_num: Optional[int] = None
    position: Optional[int] = None
    section_id: Optional[str] = None
    rendered_text: str = ""  # Plain text representation of the formula
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.formula_id:
            # Generate a unique ID based on content
            self.formula_id = generate_hash(self.content)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'formula_id': self.formula_id,
            'content': self.content,
            'is_inline': self.is_inline,
            'page_num': self.page_num,
            'position': self.position,
            'section_id': self.section_id,
            'rendered_text': self.rendered_text,
            'metadata': self.metadata
        }

@dataclass
class CodeBlockData:
    """Structured representation of a code block."""
    content: str
    language: str = ""
    code_id: str = ""
    page_num: Optional[int] = None
    position: Optional[int] = None
    section_id: Optional[str] = None
    caption: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.code_id:
            # Generate a unique ID based on content
            self.code_id = generate_hash(self.content)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'code_id': self.code_id,
            'content': self.content,
            'language': self.language,
            'page_num': self.page_num,
            'position': self.position,
            'section_id': self.section_id,
            'caption': self.caption,
            'metadata': self.metadata
        }

@dataclass
class ReferenceData:
    """Structured representation of a citation or reference."""
    text: str
    reference_id: str = ""
    citation_key: str = ""
    page_num: Optional[int] = None
    position: Optional[int] = None
    section_id: Optional[str] = None
    reference_type: str = ""  # e.g., article, book, web
    authors: List[str] = field(default_factory=list)
    year: Optional[int] = None
    title: str = ""
    source: str = ""
    doi: str = ""
    url: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.reference_id:
            # Generate a unique ID based on text
            self.reference_id = generate_hash(self.text)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'reference_id': self.reference_id,
            'text': self.text,
            'citation_key': self.citation_key,
            'page_num': self.page_num,
            'position': self.position,
            'section_id': self.section_id,
            'reference_type': self.reference_type,
            'authors': self.authors,
            'year': self.year,
            'title': self.title,
            'source': self.source,
            'doi': self.doi,
            'url': self.url,
            'metadata': self.metadata
        }

@dataclass
class LinkData:
    """Structured representation of a hyperlink."""
    text: str
    url: str
    link_id: str = ""
    page_num: Optional[int] = None
    position: Optional[int] = None
    section_id: Optional[str] = None
    link_type: str = "external"  # external, internal, anchor
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.link_id:
            # Generate a unique ID based on content
            content_str = f"{self.text}|{self.url}"
            self.link_id = generate_hash(content_str)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'link_id': self.link_id,
            'text': self.text,
            'url': self.url,
            'page_num': self.page_num,
            'position': self.position,
            'section_id': self.section_id,
            'link_type': self.link_type,
            'metadata': self.metadata
        }

@dataclass
class Section:
    """Representation of a document section."""
    title: str
    content: str
    section_id: str = ""
    level: int = 1
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    parent_id: Optional[str] = None
    path: List[str] = field(default_factory=list)  # Hierarchical path
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.section_id:
            # Generate ID using title and first 100 chars of content
            content_sample = self.content[:100] if self.content else ""
            id_text = f"{self.title}|{content_sample}"
            self.section_id = generate_hash(id_text)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'section_id': self.section_id,
            'title': self.title,
            'content': self.content,
            'level': self.level,
            'page_start': self.page_start,
            'page_end': self.page_end,
            'parent_id': self.parent_id,
            'path': self.path,
            'metadata': self.metadata,
            'content_length': len(self.content) if self.content else 0
        }

@dataclass
class ProcessedDocument:
    """Container for processed document data."""
    source: str
    doc_id: str = ""
    content: List[Section] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    tables: List[TableData] = field(default_factory=list)
    images: List[ImageData] = field(default_factory=list)
    formulas: List[FormulaData] = field(default_factory=list)
    code_blocks: List[CodeBlockData] = field(default_factory=list)
    references: List[ReferenceData] = field(default_factory=list)
    links: List[LinkData] = field(default_factory=list)

    def __post_init__(self):
        if not self.doc_id:
            self.doc_id = generate_hash(self.source)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'doc_id': self.doc_id,
            'source': self.source,
            'metadata': self.metadata,
            'content': [section.to_dict() for section in self.content],
            'tables': [table.to_dict() for table in self.tables],
            'images': [image.to_dict() for image in self.images],
            'formulas': [formula.to_dict() for formula in self.formulas],
            'code_blocks': [code_block.to_dict() for code_block in self.code_blocks],
            'references': [reference.to_dict() for reference in self.references],
            'links': [link.to_dict() for link in self.links],
        }

    def save_json(self, output_path: str) -> None:
        """Save processed document data as JSON."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    def save_images(self, output_dir: str) -> None:
        """Save all images to a directory."""
        ensure_directory(output_dir)
        for img in self.images:
            try:
                with open(os.path.join(output_dir, f"{img.image_id}.{img.extension}"), 'wb') as f:
                    f.write(img.data)
            except Exception as e:
                logger.error(f"Error saving image {img.image_id}: {e}")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProcessedDocument':
        """Create a ProcessedDocument from a dictionary."""
        # Create basic document
        doc = cls(
            doc_id=data.get('doc_id', ''),
            source=data.get('source', ''),
            metadata=data.get('metadata', {})
        )

        # Add sections
        for section_data in data.get('content', []):
            section = Section(
                section_id=section_data.get('section_id', ''),
                title=section_data.get('title', ''),
                content=section_data.get('content', ''),
                level=section_data.get('level', 1),
                page_start=section_data.get('page_start'),
                page_end=section_data.get('page_end'),
                parent_id=section_data.get('parent_id'),
                path=section_data.get('path', []),
                metadata=section_data.get('metadata', {})
            )
            doc.content.append(section)

        # Add tables
        for table_data in data.get('tables', []):
            table = TableData(
                table_id=table_data.get('table_id', ''),
                content=table_data.get('content', []),
                headers=table_data.get('headers', []),
                page_num=table_data.get('page_num'),
                position=table_data.get('position'),
                section_id=table_data.get('section_id'),
                caption=table_data.get('caption', ''),
                summary=table_data.get('summary', ''),
                metadata=table_data.get('metadata', {})
            )
            doc.tables.append(table)

        # Add images (without binary data)
        for img_data in data.get('images', []):
            # Note: Binary data needs to be loaded separately
            img = ImageData(
                data=b'',  # Empty binary data, needs to be loaded separately
                image_id=img_data.get('image_id', ''),
                extension=img_data.get('extension', ''),
                page_num=img_data.get('page_num'),
                caption=img_data.get('caption', ''),
                alt_text=img_data.get('alt_text', ''),
                width=img_data.get('width'),
                height=img_data.get('height'),
                section_id=img_data.get('section_id'),
                ocr_text=img_data.get('ocr_text', ''),
                metadata=img_data.get('metadata', {})
            )
            doc.images.append(img)

        # Add formulas
        for formula_data in data.get('formulas', []):
            formula = FormulaData(
                formula_id=formula_data.get('formula_id', ''),
                content=formula_data.get('content', ''),
                is_inline=formula_data.get('is_inline', True),
                page_num=formula_data.get('page_num'),
                position=formula_data.get('position'),
                section_id=formula_data.get('section_id'),
                rendered_text=formula_data.get('rendered_text', ''),
                metadata=formula_data.get('metadata', {})
            )
            doc.formulas.append(formula)

        # Add code blocks
        for code_data in data.get('code_blocks', []):
            code = CodeBlockData(
                code_id=code_data.get('code_id', ''),
                content=code_data.get('content', ''),
                language=code_data.get('language', ''),
                page_num=code_data.get('page_num'),
                position=code_data.get('position'),
                section_id=code_data.get('section_id'),
                caption=code_data.get('caption', ''),
                metadata=code_data.get('metadata', {})
            )
            doc.code_blocks.append(code)

        # Add references
        for ref_data in data.get('references', []):
            ref = ReferenceData(
                reference_id=ref_data.get('reference_id', ''),
                text=ref_data.get('text', ''),
                citation_key=ref_data.get('citation_key', ''),
                page_num=ref_data.get('page_num'),
                position=ref_data.get('position'),
                section_id=ref_data.get('section_id'),
                reference_type=ref_data.get('reference_type', ''),
                authors=ref_data.get('authors', []),
                year=ref_data.get('year'),
                title=ref_data.get('title', ''),
                source=ref_data.get('source', ''),
                doi=ref_data.get('doi', ''),
                url=ref_data.get('url', ''),
                metadata=ref_data.get('metadata', {})
            )
            doc.references.append(ref)

        # Add links
        for link_data in data.get('links', []):
            link = LinkData(
                link_id=link_data.get('link_id', ''),
                text=link_data.get('text', ''),
                url=link_data.get('url', ''),
                page_num=link_data.get('page_num'),
                position=link_data.get('position'),
                section_id=link_data.get('section_id'),
                link_type=link_data.get('link_type', 'external'),
                metadata=link_data.get('metadata', {})
            )
            doc.links.append(link)

        return doc


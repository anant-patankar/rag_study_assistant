# Study Assistant
## Retrieval-Augmented Generation (RAG)

A Retrieval-Augmented Generation (RAG) system designed to serve as a personalized study assistant. This system processes various document formats, chunks them effectively, generates embeddings, and retrieves relevant information based on user queries to enhance academic learning and research.

> **NOTE**: This initial release of the study assistant serves as a functional foundation for our RAG system. While it can process documents and respond to most queries effectively, users should be aware of its current limitations. Like all LLM-based systems, it may occasionally generate incorrect information or hallucinate content not found in the provided documents. You may also encounter bugs or unexpected behaviors in certain scenarios. I recommend verifying important information, especially for critical applications. I am actively working on improving accuracy, reliability, and performance in upcoming versions, and I welcome your feedback to help shape future improvements. This project represents an evolving system that will continue to be refined based on real-world usage patterns and identified issues.

## Features

- **Multi-format document processing**: Support for PDF, EPUB, TXT, DOCX, HTML, Markdown, CSV, JSON, XML, and PowerPoint files
- **Intelligent chunking**: Content-aware document chunking with various strategies (fixed, semantic, hybrid)
- **Embedding generation**: Converts text chunks to vector representations using state-of-the-art embedding models
- **Vector storage**: Efficient storage and retrieval of document vectors with FAISS and HNSWLIB
- **Query processing**: Smart query handling with hybrid search capabilities
- **LLM integration**: Uses Ollama for generating responses

---
---
## **About the Author**

I'm Anant Patankar, a Data Science Consultant with over 4.5 years of experience in developing and deploying end-to-end AI/ML solutions. I built this RAG system to demonstrate practical implementations of retrieval-augmented generation techniques for educational contexts.

- 💼 Currently seeking opportunities in generative AI development
- 🔗 [LinkedIn Profile](https://www.linkedin.com/in/anant-patankar/)
- 📧 Contact: patankar.anant123@gmail.com
<!-- - 🌐 Portfolio: [neuralnetnook.com](https://neuralnetnook.com/) -->

With expertise in Generative AI, NLP, Computer Vision, and Time Series Forecasting, I've built and deployed RAG systems using embeddings, vectorization, and LLM-powered applications with advanced prompt engineering techniques. I welcome discussions about this project, generative AI advancements, or potential collaboration opportunities.

## Technical Decisions & Learnings

Throughout this project, I made several key architectural decisions worth highlighting:

1. **Choice of embedding models**: Selected all-MiniLM-L6-v2 after testing multiple models for the optimal balance between retrieval accuracy and performance.

2. **Hybrid retrieval strategy**: Implemented a combination of dense vector similarity and sparse keyword matching to handle both semantic understanding and specific terminology.

3. **Maximum Marginal Relevance**: Added MMR reranking to reduce redundancy in retrieved contexts, addressing a common issue in many RAG implementations.

4. **Challenges overcome**: Solved memory issues with large documents by implementing streaming processing and aggressive chunk caching.

---
---

## Project Structure

```
.
├── data                # Directory for storing your documents and data
├── demo.py             # Demo script for trying out the system
├── README.md
├── requirements.txt
└── src
    ├── config          # Configuration classes and data models
    │   ├── config.py
    │   ├── data_models.py
    │   ├── doc_process_config.py
    │   └── __init__.py
    ├── __init__.py
    ├── rag_system      # Core RAG components
    │   ├── document_chunker.py     # Splitting documents into chunks
    │   ├── document_processor.py   # Document processing
    │   ├── embedding_generator.py  # Vector embedding generation
    │   ├── __init__.py
    │   ├── rag_system.py           # Main system integration
    │   └── vector_store.py         # Vector database operations
    └── utils
        ├── helpers.py              # Utility functions
        └── __init__.py
```

## Applied Generative AI Examples

### Example 1: Complex Query Handling

```python
# The system can handle nuanced queries that require synthesizing information
# from multiple sections of a document
result = rag.query("Compare the advantages and limitations of transformer models as discussed in chapters 3 and 7")
```

### Example 2: Multimodal Content Processing

```python
# The system can process documents containing text, tables, and images
# and reference them appropriately in generated responses
rag.process_document("research_paper_with_diagrams.pdf")
result = rag.query("Summarize the experimental results shown in Figure 3 and Table 2")
```


## System Requirements

This system runs local LLM models using Ollama and requires sufficient hardware resources:

### Minimum Requirements
- **CPU**: 4+ cores
- **RAM**: 8GB (16GB recommended for larger documents)
- **Storage**: 10GB free space for application and model storage
- **GPU**: Not required, but CUDA-compatible GPU highly recommended for faster embedding generation
- **Operating System**: Linux, macOS, or Windows 10/11

### Recommended for Optimal Performance
- **CPU**: 8+ cores
- **RAM**: 32GB
- **Storage**: SSD with 20GB+ free space
- **GPU**: CUDA-compatible GPU with 8GB+ VRAM
- **Operating System**: Ubuntu 20.04+ or macOS 12+

### Software Prerequisites
- Python 3.8 or higher
- [Ollama](https://ollama.ai/) installed and running locally
- CUDA Toolkit 11.4+ (if using GPU acceleration)

## Model Selection for study assistant Use

This RAG system is specifically designed as an study assistant. By default, it uses the `deepseek-r1:1.5b` model from Ollama, which offers a good balance between performance and resource requirements. However, you can configure the system to use different models based on your hardware capabilities and specific needs:

| Model | Size | Strengths | Min RAM | Recommended For |
|-------|------|-----------|---------|-----------------|
| deepseek-r1:1.5b | 1.5B | Fast, efficient | 4GB | Basic study tasks, resource-constrained systems |
| llama3:8b | 8B | Better comprehension | 16GB | General academic research, better understanding |
| mistral:7b | 7B | Strong reasoning | 16GB | Technical subjects, nuanced explanations |
| llama3:70b | 70B | Most comprehensive | 32GB+ and GPU | Deep research, complex academic topics |

You can change the model in the `config.py` file by modifying the `model_name` parameter in the `GenerationConfig` class. (Simpler configuration methods will be added in future).

## Getting Started

1. Install Ollama following the instructions at [https://ollama.ai/download](https://ollama.ai/download)

2. Pull your preferred model (starting with the default):
```bash
ollama pull deepseek-r1:1.5b
```

3. (Optional) Pull alternative models based on your academic needs:
```bash
# For general academic research
ollama pull llama3:8b

# For technical subjects
ollama pull mistral:7b

# For advanced research (requires significant resources)
ollama pull llama3:70b
```

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

4. Configure the system by modifying the configuration in `config.py` or by using the predefined configurations.

5. Initialize and use the RAG system:
```python
from src.config import RagSystemConfig
from src.rag_system import RagSystem

# Create with default configuration
config = RagSystemConfig()
rag = RagSystem(config)

# Process documents
rag.process_document("path/to/your/document.pdf")

# Query the system
result = rag.query("Your question about the document?")
print(result.answer)
```

## To-do List:
> Note: Completed task are marked with white tick-mark (&#10004;) and Pendig task are marked with white cross (&#10008;) and for tasks which are iterative are marked with &#10227;

> **NOTE**: This initial implementation of the study assistant is operational but still evolving. You may encounter occasional issues. We welcome bug reports and feature suggestions to improve future versions.

### Source Code Improvemets:
- &#10004; Standardized variable naming conventions and comments across all modules
- &#10004; Refactored document processing to improve error handling
- &#10004; Optimized embedding batch size for better GPU utilization
- &#10004; Fixed inconsistent section hierarchy detection
- &#10004; Improved table extraction accuracy for complex layouts
- &#10004; Added better error recovery in PDF processing
- &#10004; Normalized metadata field names for consistency
- &#10004; Restructured vector store initialization for clarity
- &#10004; Enhanced MMR implementation for better result diversity
- &#10004; Fixed memory leak in image processing pipeline
- &#10227; Fixing checking for process or loggical errors and doing modifications for it

### Core Functionalities:
- &#10008; Add proper error handling and strong recovery mechanisms
- &#10008; Adding document update/ deletion functionality
- &#10008; Adding support for more file formats (e.g. audio transcriptions, image OCR)
- &#10008; Enhance multilingual support (will be decided after feasibility checking)
- &#10008; Enhanced Prompts to get more accurate results
- &#10008; Creating installaion scripts (If needed)
- &#10008; Implementation of monitoring and logging solutions

### Optimizations (Improving Performance):
- &#10008; Optimization memory usage for large document collections
- &#10008; Mechanism for storing frequent results of common searches so they load quicker next time
- &#10008; Mechanism to embedding model based on document types.
- &#10008; Enhancing retrieval process to get more accurate results
- &#10008; Adding mechanisms to check if document is correctly processed and added to index

### Evaluation Metrix and Testing:
- &#10008; Implementing testing mechanism for components to ensure seamless performance
- &#10008; Implementing testing mechanism for malformed or corrupt files
- &#10008; Implement test for system behaviour during partial failures
- &#10008; Adding Answer Evaluation Framework to measure quality and effectiveness of system response

### Documentation:
- &#10008; Complete project documentation
- &#10008; Creating detailed installation guide
- &#10008; Creating usage examples and guide
- &#10008; Documenting Configuration Options with reasoning of configuration parameters
- &#10008; Adding Process flow diagram
- &#10008; Creating deployment guide for various environments


## Future Development Roadmap

While the current implementation demonstrates core RAG capabilities, I'm planning several advancements:

- **Improved hallucination detection**: Implementing a reference reconciliation system that validates generated content against source material in real-time

- **Fine-tuned embedding models**: Developing domain-specific embeddings for academic content to improve retrieval precision

- **Self-healing error recovery**: Building a system that can detect and recover from processing failures without user intervention

- **Adaptive chunking**: Implementing content-aware chunking that adjusts strategies based on document type and content density

## Technologies & Concepts Demonstrated

- Large Language Models (LLMs)
- Retrieval-Augmented Generation (RAG)
- Vector Embeddings & Semantic Search
- Ollama Integration
- Knowledge Management
- Generative AI Applications
- Natural Language Processing (NLP)
- Educational Technology
- Document Processing & Analysis
- Context-Aware Response Generation


## License

[GNU General Public License v3.0](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

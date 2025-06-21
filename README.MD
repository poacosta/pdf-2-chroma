# PDF2Chroma

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-1.0.0+-green.svg)](https://www.trychroma.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready Python script that transforms PDF document collections into locally-stored, semantically searchable
vector databases using ChromaDB's persistent storage. This script bridges the gap between static document repositories
and intelligent information retrieval systems.

## The Vector Database Transformation Pipeline

Modern knowledge management faces a fundamental challenge: converting unstructured document content into computationally
accessible representations. This script implements a pipeline that addresses three critical transformation stages:

**Stage 1: Document Deconstruction**

- Intelligent PDF text extraction with structural preservation
- Page-level content organization and metadata capture
- Quality filtering to eliminate processing artifacts

**Stage 2: Semantic Chunking**

- Boundary-aware text segmentation that respects semantic coherence
- Configurable overlap strategies for context preservation
- Content-based unique identifier generation

**Stage 3: Vector Storage**

- Embedding generation using state-of-the-art transformer models
- Persistent local storage in ChromaDB's optimized format
- Hierarchical metadata preservation for advanced filtering

**Mechanical Insights:** The script leverages ChromaDB's embedded database architecture, storing all vector data locally
in a self-contained directory structure. This eliminates external dependencies while providing enterprise-grade search
capabilities.

**Implications:** By maintaining complete local control over document embeddings, organizations can implement RAG (
Retrieval Augmented Generation) systems without external API dependencies or data privacy concerns.

## Local Storage Architecture

The script creates a persistent ChromaDB instance that stores all document vectors, metadata, and search indices in your
local filesystem:

```
pdf-2-chroma/
├── docs/                      # Source PDF directory
├── knowledge_base/            # ChromaDB persistent storage
│   ├── chroma.sqlite3         # Vector database file
└── main.py                    # This script
```

**Storage Characteristics:**

- **Self-Contained:** All data remains on your local machine
- **Persistent:** Survives script restarts and system reboots
- **Portable:** Database directory can be copied between systems
- **Scalable:** Handles collections from hundreds to millions of chunks

## Installation and Setup

### Dependencies

```bash
# Core requirements
pip install chromadb pdfplumber

# Optional: OpenAI embeddings
pip install openai
```

## Script Configuration and Usage

### Basic Document Processing

```python
# Initialize the document processor
loader = ChromaDBDocumentLoader(
    persist_directory="./my_knowledge_base",  # Local storage location
    embedding_model="all-MiniLM-L6-v2"  # Local embedding model
)

# Process all PDFs in a directory
results = loader.load_pdfs_from_folder(
    folder_path="./company_documents",
    collection_name="corporate_kb"
)
```

**What Happens During Processing:**

1. **PDF Discovery:** Script scans the specified directory for PDF files
2. **Text Extraction:** Each PDF is processed page-by-page using pdfplumber
3. **Intelligent Chunking:** Text is segmented with semantic boundary detection
4. **Embedding Generation:** Chunks are converted to vectors using the specified model
5. **Local Storage:** All data is persisted to the ChromaDB directory

### Advanced Configuration Options

#### Embedding Model Selection

The script supports multiple embedding strategies, each with distinct performance characteristics:

**Local Sentence Transformers:**

```python
# Balanced performance and quality
embedding_model = "all-MiniLM-L6-v2"  # 384 dimensions, fast processing

# Superior semantic understanding
embedding_model = "all-mpnet-base-v2"  # 768 dimensions, slower but higher quality
```

**OpenAI Cloud Embeddings:**

```python
loader = ChromaDBDocumentLoader(
    use_openai_embeddings=True,
    openai_model="text-embedding-3-small",  # Cost-effective cloud processing
    # Requires OPENAI_API_KEY environment variable
)
```

**Insights:** Local models provide complete data privacy and eliminate API costs, while cloud models offer superior
semantic understanding at the expense of external dependencies.

#### Chunking Strategy Optimization

```python
loader = ChromaDBDocumentLoader(
    chunk_size=1200,  # Larger contexts for complex documents
    chunk_overlap=200,  # Enhanced context preservation
    min_chunk_size=100  # Filter out fragmentary content
)
```

**Chunking Algorithm Details:**

- **Boundary Detection:** Prioritizes sentence endings over word boundaries
- **Overlap Strategy:** Maintains semantic continuity between adjacent chunks
- **Size Optimization:** Balances context preservation with computational efficiency

### Local Database Querying

Once documents are loaded, the local ChromaDB instance enables sophisticated search operations:

```python
# Semantic similarity search
results = loader.search_documents(
    collection_name="corporate_kb",
    query="quarterly financial performance metrics",
    n_results=5
)

# Metadata-filtered search
filtered_results = loader.search_documents(
    collection_name="corporate_kb",
    query="product specifications",
    filter_metadata={"department": "engineering"}
)
```

**Query Processing Flow:**

1. **Query Embedding:** Search text is converted to vector representation
2. **Similarity Calculation:** ChromaDB computes cosine distances across all stored vectors
3. **Ranking:** Results are sorted by semantic relevance
4. **Metadata Enrichment:** Source documents and page numbers are included

## Script Architecture Deep Dive

### PDF Processing Pipeline

The document processing pipeline implements several approaches to common PDF extraction challenges:

**Text Extraction Challenges and Solutions:**

- **Challenge:** PDF formatting artifacts and layout inconsistencies
    - **Solution:** pdfplumber's advanced parsing with post-processing cleanup
- **Challenge:** Maintaining document structure and page relationships
    - **Solution:** Hierarchical metadata preservation across file, page, and chunk levels

**Chunking Innovation:**
Traditional PDF processors often fragment content at arbitrary character boundaries, destroying semantic coherence. This
script implements boundary-aware chunking:

```python
def chunk_text(self, text: str, base_metadata: Dict[str, Any]) -> List[DocumentChunk]:
# Intelligent boundary detection hierarchy:
# 1. Paragraph breaks (\n\n) - Primary preference
# 2. Sentence endings (., !, ?) - Secondary preference  
# 3. Word boundaries - Fallback mechanism
# 4. Character limits - Hard constraint
```

**Insights:** By respecting natural language boundaries, the chunking algorithm preserves semantic
relationships that would be lost in naive character-based splitting approaches.

### Local Database Persistence

ChromaDB's embedded architecture provides enterprise-grade persistence without external infrastructure:

**Storage Technology Stack:**

- **SQLite Backend:** ACID-compliant relational storage for metadata
- **Vector Indices:** Optimized similarity search structures
- **File-Based Persistence:** Self-contained storage requiring no database servers

**Data Integrity Mechanisms:**

- **Unique Chunk IDs:** Content-based hashing prevents duplicates
- **Incremental Updates:** Skip processing for previously loaded documents
- **Error Recovery:** Graceful handling of corrupted or protected PDFs

### Embedding Model Integration

The script abstracts embedding generation to support multiple model architectures:

**Local Model Advantages:**

- Complete data privacy and offline operation
- No usage-dependent costs or rate limits
- Deterministic results for reproducible experiments

**Cloud Model Advantages:**

- Superior semantic understanding from large-scale training
- Regular model updates and improvements
- Reduced local computational requirements

**Performance Comparison:**

```
Model                    | Dimensions | Processing Speed | Semantic Quality
------------------------|------------|------------------|------------------
all-MiniLM-L6-v2        | 384        | ~1000 docs/min  | Good
all-mpnet-base-v2       | 768        | ~400 docs/min   | Excellent  
text-embedding-3-small  | 1536       | ~600 docs/min   | Excellent
text-embedding-3-large  | 3072       | ~200 docs/min   | Superior
```

## Running the Script

### Command Line Execution

```bash
# Basic usage
python main.py

# With custom configuration
python main.py --source-dir ./docs --db-dir ./knowledge_base --model all-mpnet-base-v2
```

## Security and Privacy Considerations

The local storage architecture provides several security advantages:

**Data Sovereignty:**

- All document content remains on local infrastructure
- No external API calls for basic embedding models
- Complete control over access patterns and usage logs

**Privacy Preservation:**

- Document text never transmitted to external services (with local models)
- Embedding vectors stored locally without external dependencies
- Metadata filtering enables sensitive document segregation

**Compliance Benefits:**

- Supports GDPR, HIPAA, and other data protection requirements
- Enables air-gapped deployment scenarios
- Facilitates audit trails through local logging

## Troubleshooting Common Issues

### PDF Processing Errors

* **Issue:** Some PDFs fail to process
  * **Solution:** Check file permissions and PDF encryption status

* **Issue:** Extracted text contains formatting artifacts  
  * **Solution:** Adjust `min_chunk_size` parameter to filter fragments

### Database Performance

* **Issue:** Slow query responses on large collections
  * **Solution:** Implement collection partitioning or increase SSD storage

* **Issue:** High memory usage during processing
  * **Solution:** Reduce `batch_size` parameter for available RAM

### Embedding Model Selection

* **Issue:** Poor search relevance with default model
  * **Solution:** Upgrade to `all-mpnet-base-v2` for better semantic understanding

* **Issue:** OpenAI API rate limits
  * **Solution:** Switch to local models for unlimited processing

## License

MIT License - see [LICENSE](LICENSE) file for details.

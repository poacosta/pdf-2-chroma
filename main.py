import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import chromadb
import pdfplumber
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
from chromadb.utils import embedding_functions


@dataclass
class DocumentChunk:
    """Represents a chunk of text from a document."""
    text: str
    metadata: Dict[str, Any]
    chunk_id: str
    source_file: str
    page_number: Optional[int] = None
    chunk_index: int = 0


class PDFProcessor:
    """
    Advanced PDF processing with intelligent text extraction and chunking.
    """

    def __init__(
            self,
            chunk_size: int = 1000,
            chunk_overlap: int = 200,
            min_chunk_size: int = 100
    ):
        """
        Initialize PDF processor.

        Args:
            chunk_size: Target size for text chunks (characters)
            chunk_overlap: Overlap between consecutive chunks
            min_chunk_size: Minimum chunk size to keep
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.logger = logging.getLogger(__name__)

    def extract_text_from_pdf(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """
        Extract text from PDF with page-level metadata.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of page dictionaries with text and metadata
        """
        pages = []

        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text()

                    if text and text.strip():
                        # Clean extracted text
                        cleaned_text = self._clean_text(text)

                        if len(cleaned_text) >= self.min_chunk_size:
                            page_info = {
                                'text': cleaned_text,
                                'page_number': page_num,
                                'total_pages': len(pdf.pages),
                                'file_path': str(pdf_path),
                                'file_name': pdf_path.name,
                                'file_size': pdf_path.stat().st_size,
                                'modification_time': datetime.fromtimestamp(pdf_path.stat().st_mtime).isoformat()
                            }
                            pages.append(page_info)

        except Exception as e:
            self.logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
            raise

        return pages

    def _clean_text(self, text: str) -> str:
        """Clean extracted text by removing excessive whitespace and fixing common issues."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove page numbers and headers/footers (basic heuristic)
        text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
        # Remove excessive newlines
        text = re.sub(r'\n+', '\n', text)
        return text.strip()

    def chunk_text(self, text: str, base_metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """
        Split text into overlapping chunks with metadata preservation.

        Args:
            text: Text to chunk
            base_metadata: Base metadata to attach to each chunk

        Returns:
            List of DocumentChunk objects
        """
        if len(text) <= self.chunk_size:
            chunk_id = self._generate_chunk_id(text, base_metadata['file_name'], 0)
            return [DocumentChunk(
                text=text,
                metadata=base_metadata.copy(),
                chunk_id=chunk_id,
                source_file=base_metadata['file_name'],
                page_number=base_metadata.get('page_number'),
                chunk_index=0
            )]

        chunks = []
        start = 0
        chunk_index = 0

        while start < len(text):
            end = start + self.chunk_size

            # Try to find a good breaking point (sentence or paragraph)
            if end < len(text):
                # Look for sentence boundaries
                sentence_breaks = ['.', '!', '?', '\n\n']
                best_break = end

                for i in range(end - 100, end + 100):
                    if i >= len(text):
                        break
                    if text[i] in sentence_breaks:
                        best_break = i + 1
                        break

                end = best_break

            chunk_text = text[start:end].strip()

            if len(chunk_text) >= self.min_chunk_size:
                chunk_metadata = base_metadata.copy()
                chunk_metadata.update({
                    'chunk_index': chunk_index,
                    'chunk_size': len(chunk_text),
                    'start_char': start,
                    'end_char': end
                })

                chunk_id = self._generate_chunk_id(chunk_text, base_metadata['file_name'], chunk_index)

                chunks.append(DocumentChunk(
                    text=chunk_text,
                    metadata=chunk_metadata,
                    chunk_id=chunk_id,
                    source_file=base_metadata['file_name'],
                    page_number=base_metadata.get('page_number'),
                    chunk_index=chunk_index
                ))

                chunk_index += 1

            start = end - self.chunk_overlap if end < len(text) else len(text)

        return chunks

    def _generate_chunk_id(self, text: str, filename: str, chunk_index: int) -> str:
        """Generate unique chunk ID based on content and metadata."""
        content_hash = hashlib.md5(f"{filename}_{chunk_index}_{text[:100]}".encode()).hexdigest()[:8]
        return f"{Path(filename).stem}_{chunk_index}_{content_hash}"


class ChromaDBDocumentLoader:
    """
    Production-ready ChromaDB document loader with PDF support.
    """

    def __init__(
            self,
            persist_directory: str = "./chroma_documents_db",
            embedding_model: str = "all-MiniLM-L6-v2",
            use_openai_embeddings: bool = False,
            openai_model: str = "text-embedding-3-small",
            chunk_size: int = 1000,
            chunk_overlap: int = 200,
            tenant: str = DEFAULT_TENANT,
            database: str = DEFAULT_DATABASE
    ):
        """
        Initialize the document loader.

        Args:
            persist_directory: Directory for persistent storage
            embedding_model: Model for embeddings
            use_openai_embeddings: Whether to use OpenAI embeddings
            openai_model: OpenAI model name
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            tenant: ChromaDB tenant
            database: ChromaDB database
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(),
            tenant=tenant,
            database=database
        )

        # Configure embedding function
        if use_openai_embeddings:
            if not os.getenv("OPENAI_API_KEY"):
                raise ValueError("OPENAI_API_KEY environment variable required")
            self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                model_name=openai_model,
                api_key=os.getenv("OPENAI_API_KEY")
            )
        else:
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=embedding_model
            )

        # Initialize PDF processor
        self.pdf_processor = PDFProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def load_pdfs_from_folder(
            self,
            folder_path: str,
            collection_name: str,
            file_pattern: str = "*.pdf",
            batch_size: int = 100,
            skip_existing: bool = True
    ) -> Dict[str, Any]:
        """
        Load all PDF files from a folder into ChromaDB.

        Args:
            folder_path: Path to folder containing PDFs
            collection_name: Name of the ChromaDB collection
            file_pattern: Pattern to match PDF files
            batch_size: Number of chunks to process per batch
            skip_existing: Skip files that are already processed

        Returns:
            Summary of processing results
        """
        folder = Path(folder_path)
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        # Get or create collection
        collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={
                "description": f"Document collection from {folder_path}",
                "created_at": datetime.now().isoformat(),
                "source_folder": str(folder)
            }
        )

        # Find PDF files
        pdf_files = list(folder.glob(file_pattern))
        if not pdf_files:
            self.logger.warning(f"No PDF files found in {folder_path}")
            return {"processed_files": 0, "total_chunks": 0, "errors": []}

        processing_results = {
            "processed_files": 0,
            "total_chunks": 0,
            "errors": [],
            "skipped_files": []
        }

        # Get existing document IDs to avoid duplicates
        existing_ids = set()
        if skip_existing:
            try:
                existing_data = collection.get(include=["metadatas"])
                existing_ids = {
                    metadata.get("file_name", "") for metadata in existing_data["metadatas"]
                }
            except Exception as e:
                self.logger.warning(f"Could not check existing documents: {e}")

        for pdf_file in pdf_files:
            try:
                if skip_existing and pdf_file.name in existing_ids:
                    self.logger.info(f"Skipping already processed file: {pdf_file.name}")
                    processing_results["skipped_files"].append(pdf_file.name)
                    continue

                self.logger.info(f"Processing: {pdf_file.name}")

                # Extract text from PDF
                pages = self.pdf_processor.extract_text_from_pdf(pdf_file)

                if not pages:
                    self.logger.warning(f"No text extracted from {pdf_file.name}")
                    continue

                # Process all chunks for this file
                all_chunks = []
                for page in pages:
                    chunks = self.pdf_processor.chunk_text(page['text'], page)
                    all_chunks.extend(chunks)

                if all_chunks:
                    # Add chunks to collection in batches
                    self._add_chunks_in_batches(collection, all_chunks, batch_size)
                    processing_results["total_chunks"] += len(all_chunks)
                    processing_results["processed_files"] += 1

                    self.logger.info(f"Added {len(all_chunks)} chunks from {pdf_file.name}")

            except Exception as e:
                error_msg = f"Error processing {pdf_file.name}: {str(e)}"
                self.logger.error(error_msg)
                processing_results["errors"].append(error_msg)

        self.logger.info(f"Processing complete: {processing_results}")
        return processing_results

    def _add_chunks_in_batches(
            self,
            collection: chromadb.Collection,
            chunks: List[DocumentChunk],
            batch_size: int
    ) -> None:
        """Add document chunks to collection in batches."""
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]

            documents = [chunk.text for chunk in batch]
            metadatas = [chunk.metadata for chunk in batch]
            ids = [chunk.chunk_id for chunk in batch]

            try:
                collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
            except Exception as e:
                self.logger.error(f"Error adding batch {i // batch_size + 1}: {str(e)}")
                raise

    def search_documents(
            self,
            collection_name: str,
            query: str,
            n_results: int = 5,
            filter_metadata: Optional[Dict[str, Any]] = None,
            include_source_info: bool = True
    ) -> Dict[str, Any]:
        """
        Search documents with enhanced result formatting.

        Args:
            collection_name: Collection to search
            query: Search query
            n_results: Number of results
            filter_metadata: Metadata filters
            include_source_info: Include source file information

        Returns:
            Formatted search results
        """
        collection = self.client.get_collection(collection_name)

        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            where=filter_metadata,
            include=["documents", "metadatas", "distances"]
        )

        # Format results for better readability
        formatted_results = []
        for i, doc in enumerate(results["documents"][0]):
            metadata = results["metadatas"][0][i]
            distance = results["distances"][0][i]

            result = {
                "content": doc,
                "relevance_score": 1 - distance,  # Convert distance to similarity score
                "source_file": metadata.get("file_name", "Unknown"),
                "page_number": metadata.get("page_number"),
                "chunk_index": metadata.get("chunk_index"),
            }

            if include_source_info:
                result.update({
                    "file_size": metadata.get("file_size"),
                    "modification_time": metadata.get("modification_time"),
                    "chunk_size": metadata.get("chunk_size")
                })

            formatted_results.append(result)

        return {
            "query": query,
            "results": formatted_results,
            "total_found": len(formatted_results)
        }

    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Get comprehensive collection information."""
        collection = self.client.get_collection(collection_name)

        # Get sample of documents to analyze
        sample_data = collection.get(limit=100, include=["metadatas"])

        # Analyze file distribution
        file_stats = {}
        for metadata in sample_data["metadatas"]:
            filename = metadata.get("file_name", "Unknown")
            file_stats[filename] = file_stats.get(filename, 0) + 1

        return {
            "name": collection.name,
            "total_documents": collection.count(),
            "metadata": collection.metadata,
            "unique_files": len(file_stats),
            "file_distribution": dict(sorted(file_stats.items(), key=lambda x: x[1], reverse=True)[:10])
        }

    def export_collection_metadata(self, collection_name: str, output_file: str) -> None:
        """Export collection metadata to JSON file for analysis."""
        collection = self.client.get_collection(collection_name)

        all_data = collection.get(include=["metadatas"])

        export_data = {
            "collection_name": collection_name,
            "export_timestamp": datetime.now().isoformat(),
            "total_documents": len(all_data["ids"]),
            "metadata": all_data["metadatas"]
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Metadata exported to {output_file}")


# Example usage
def main():
    """Demonstration of a PDF document loading into ChromaDB."""

    # Initialize the document loader
    loader = ChromaDBDocumentLoader(
        persist_directory="./knowledge_base",
        embedding_model="all-MiniLM-L6-v2",  # Use "all-mpnet-base-v2" for better quality
        chunk_size=800,
        chunk_overlap=100
    )

    # Load PDFs from folder
    pdf_folder = "./docs"
    collection_name = "borges_stories"

    try:
        # Process all PDFs in the folder
        results = loader.load_pdfs_from_folder(
            folder_path=pdf_folder,
            collection_name=collection_name,
            batch_size=50,
            skip_existing=True
        )

        print(f"Processing Results:")
        print(f"  Files processed: {results['processed_files']}")
        print(f"  Total chunks created: {results['total_chunks']}")
        print(f"  Files skipped: {len(results['skipped_files'])}")
        print(f"  Errors: {len(results['errors'])}")

        if results['errors']:
            print("Errors encountered:")
            for error in results['errors']:
                print(f"  - {error}")

        # Get collection information
        collection_info = loader.get_collection_info(collection_name)
        print(f"\nCollection Info:")
        print(f"  Total documents: {collection_info['total_documents']}")
        print(f"  Unique files: {collection_info['unique_files']}")
        print(f"  Top files by chunk count:")
        for filename, count in list(collection_info['file_distribution'].items())[:5]:
            print(f"    {filename}: {count} chunks")

        # Example search
        search_results = loader.search_documents(
            collection_name=collection_name,
            query="Who is Emma Zunz?",
            n_results=3,
            filter_metadata=None
        )

        print(f"\nSearch Results for: '{search_results['query']}'")
        for i, result in enumerate(search_results['results'], 1):
            print(f"  {i}. [{result['relevance_score']:.3f}] {result['source_file']} (Page {result['page_number']})")
            print(f"     {result['content'][:200]}...")

        # Export metadata for analysis
        loader.export_collection_metadata(collection_name, f"{collection_name}_metadata.json")

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()

import pandas as pd
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter



class DataProcessor:
    """Data processing class."""

    def __init__(self, config: Dict[str, Any], logger=None):
        self.config = config
        self.logger = logger
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config['data']['chunk_size'],
            chunk_overlap=config['data']['chunk_overlap'],
            separators=["\n\n", "\n", "。", "；", "，", " ", ""],
            length_function=len
        )

    def preprocess_data(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Preprocess data and merge relevant columns."""
        documents = []
        text_columns = self.config['data']['text_columns']

        for idx, row in df.iterrows():
            # Merge specified columns
            text_parts = []
            for col in text_columns:
                if col in row and pd.notna(row[col]):
                    text_parts.append(str(row[col]))

            if text_parts:
                # Combine all text
                full_text = "\n".join(text_parts)
                metadata = row.to_dict()

                documents.append({
                    "id": idx,
                    "text": full_text,
                    "metadata": metadata
                })

        if self.logger:
            self.logger.info(f"Preprocessing complete. Generated {len(documents)} documents.")

        return documents

    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Chunk documents into smaller text pieces."""
        chunks = []

        for doc in documents:
            text = doc["text"]
            metadata = doc["metadata"]

            # Split text
            text_chunks = self.text_splitter.split_text(text)

            for i, chunk in enumerate(text_chunks):
                chunk_metadata = metadata.copy()
                chunk_metadata["chunk_id"] = i
                chunk_metadata["total_chunks"] = len(text_chunks)
                chunk_metadata["source_id"] = doc["id"]

                chunks.append({
                    "text": chunk,
                    "metadata": chunk_metadata
                })

        if self.logger:
            self.logger.info(f"Chunking complete. Generated {len(chunks)} text chunks.")

        return chunks

    def process_csv(self, csv_path: str) -> List[Dict[str, Any]]:
        """Complete data processing pipeline."""
        # Load CSV
        df = pd.read_csv(csv_path, encoding='utf-8')

        if self.logger:
            self.logger.info(f"Original data shape: {df.shape}")

        # Preprocess
        documents = self.preprocess_data(df)

        # Chunk
        chunks = self.chunk_documents(documents)

        return chunks
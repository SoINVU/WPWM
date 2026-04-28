import faiss
import numpy as np
import os
from typing import List, Dict, Any, Tuple


class VectorDatabase:
    """Vector database management."""

    def __init__(self, index_path: str, dimension: int, logger=None):
        self.index_path = index_path
        self.dimension = dimension
        self.logger = logger
        self.index = None
        self.chunks = []
        self.metadata = []

    def create_index(self, embeddings: np.ndarray, chunks: List[Dict[str, Any]]):
        """Create FAISS index."""

        dimension = embeddings.shape[1]
        self.dimension = dimension

        if self.logger:
            self.logger.info(f"Creating FAISS index. Dimension: {dimension}, Data size: {len(embeddings)}")

        embeddings = embeddings.astype('float32')

        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)

        self.chunks = [chunk["text"] for chunk in chunks]
        self.metadata = [chunk["metadata"] for chunk in chunks]

        if self.logger:
            self.logger.info("FAISS index successfully created.")

    def save(self):
        """Save index and data."""
        if self.index is None:
            raise ValueError("Index not created.")

        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, f"{
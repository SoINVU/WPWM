import os
from typing import List, Dict, Any, Tuple


class RAGSystem:
    """Main class for the RAG system"""

    def __init__(self, config_path: str = "/mnt/workspace/config.yaml"):
        from utils.helpers import load_config, setup_logging, check_cuda

        # Load configuration
        self.config = load_config(config_path)

        # Set up logging
        self.logger = setup_logging(
            self.config['system']['log_dir'],
            "rag_system"
        )

        # Check CUDA availability
        cuda_info = check_cuda()
        self.logger.info(f"CUDA availability: {cuda_info}")

        # Initialize components
        self.embedding_model = None
        self.vector_db = None
        self.data_processor = None

        # Import other modules
        from data_processor import DataProcessor
        from embedding_model import QwenEmbeddingModel
        from vector_db import VectorDatabase

        self.DataProcessor = DataProcessor
        self.QwenEmbeddingModel = QwenEmbeddingModel
        self.VectorDatabase = VectorDatabase

    def initialize_models(self):
        """Initialize models"""
        self.logger.info("Starting model initialization...")

        # Initialize embedding model
        self.embedding_model = self.QwenEmbeddingModel(
            model_path=self.config['models']['embedding']['path'],
            device=self.config['models']['embedding']['device'],
            max_length=self.config['models']['embedding']['max_length'],
            logger=self.logger
        )

        # Initialize data processor
        self.data_processor = self.DataProcessor(
            config=self.config,
            logger=self.logger
        )

        # Initialize vector database
        self.vector_db = self.VectorDatabase(
            index_path=self.config['vector_db']['index_path'],
            dimension=None,
            logger=self.logger
        )

        self.logger.info("All models initialized successfully.")

    def build_vector_database(self, csv_path: str = None):
        """Build the vector database"""
        if csv_path is None:
            csv_path = os.path.join(
                self.config['system']['workspace_dir'],
                self.config['data']['csv_path']
            )

        self.logger.info(f"Starting to build vector database, data source: {csv_path}")

        # Process data
        chunks = self.data_processor.process_csv(csv_path)

        # Extract texts
        texts = [chunk["text"] for chunk in chunks]

        # Generate text embeddings
        self.logger.info("Generating text embeddings...")
        embeddings = self.embedding_model.encode(texts, batch_size=4)

        # Create index
        self.vector_db.create_index(embeddings, chunks)

        # Save index
        self.vector_db.save()

        self.logger.info(f"Vector database built successfully, containing {len(chunks)} document chunks.")

    def load_vector_database(self):
        """Load the vector database"""
        self.logger.info("Loading vector database...")
        self.vector_db.load()
        self.logger.info("Vector database loaded successfully.")

    def search(self, query: str, top_k: int = None) -> Tuple[List[Any], List[Dict], List[float]]:
        """Search for relevant documents"""
        if top_k is None:
            top_k = self.config['rag']['top_k']

        self.logger.info(f"Search query: {query}")

        # Generate query embedding
        query_embedding = self.embedding_model.encode(query)[0]

        # Vector search
        documents, metadata_list, scores = self.vector_db.search(
            query_embedding, top_k=top_k
        )

        if not documents:
            self.logger.warning("No relevant maintenance information found.")
            return [], [], []

        return documents, metadata_list, scores
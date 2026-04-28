import argparse
from rag_system import RAGSystem


def main():
    parser = argparse.ArgumentParser(description="main_faiss_built")
    parser.add_argument("--csv", type=str,
                        default="/mnt/workspace/data/maintenance_suggestions.csv",
                        help="Path to the CSV file")
    parser.add_argument("--config", type=str,
                        default="/mnt/workspace/config.yaml",
                        help="Path to the configuration file")

    args = parser.parse_args()

    # Initialize the RAG system
    rag_system = RAGSystem(config_path=args.config)
    rag_system.initialize_models()

    # Build the vector database
    print("Starting to build the vector database...")
    rag_system.build_vector_database(args.csv)
    print("Vector database build completed!")


if __name__ == "__main__":
    main()
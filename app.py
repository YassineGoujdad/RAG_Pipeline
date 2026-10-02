from src.rag.data_loader import load_all_documents
from src.rag.embedding import EmbeddingPipeline
from src.rag.vectorstore import FaissVectorStore

if __name__ == "__main__":
    docs = load_all_documents("data")

    store = FaissVectorStore("Faiss_store")

    store.build_from_documents(docs )

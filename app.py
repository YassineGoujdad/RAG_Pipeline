from src.rag.data_loader import load_all_documents
from src.rag.embedding import EmbeddingPipeline
from src.rag.vectorstore import FaissVectorStore
from src.rag.search import RAGSearch


if __name__ == "__main__":
    docs = load_all_documents("data")

    store = FaissVectorStore("Faiss_store")

    #store.build_from_documents(docs )
    store.load()

    rag_search = RAGSearch()
    query = "what is customer success"
    
    summary = rag_search.search_and_summarize(query, top_k=3)

    print("Summary: ", summary)

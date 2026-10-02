import os 
from dotenv import load_dotenv
from src.rag.vectorstore import FaissVectorStore
from langchain_openai import ChatOpenAI

load_dotenv()

class RAGSearch:
    def __init__(self, persist_dir: str = "faiss_store", embedding_model:str = "all-MiniLM-L6-v2", llm_model: str ="gpt-4o-mini"):
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)

        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")

        if not (os.path.exists(faiss_path) and os.path.exists(meta_path) ):
            from src.rag.data_loader import load_all_documents
            docs = load_all_documents("data")
            self.vectorstore.build_from_documents(docs)
        else:
            self.vectorstore.load()

        Openai_api_key = os.getenv("OPEN_AI_KEY")
        self.llm = ChatOpenAI(Openai_api_key = Openai_api_key, model_name=llm_model)
        print(f"[INFO] OpenAI LLM initialized: {llm_model}")

    def search_and_summarize(self, query: str, top_k: int = 5)-> str:
        results = self.vectorstore.query(query, top_k = top_k)
        texts = [r["metadata"].get("text","") for r in results if r["metadata"]]

        context = "\n\n".join(texts)
        if not context:
            return "No relevant documents found."

        prompt = f""" Summarize the following context for the query: '{query}' \n\n Context: \n {context} \n \nSummary: """

        response = self.llm.invoke([prompt])
        return response.content
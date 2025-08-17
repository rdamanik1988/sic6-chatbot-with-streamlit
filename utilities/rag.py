from langgraph.graph import START, StateGraph
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain import hub
from langchain_core.documents import Document
from typing_extensions import List, TypedDict
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

# Initialize State

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

class RAG:
    def __init__(self):
        self.graph_builder = StateGraph(State).add_sequence([self.retrieve, self.generate])
        self.graph_builder.add_edge(START, "retrieve")
        self.graph = self.graph_builder.compile()
        
        # Mengganti GoogleGenerativeAIEmbeddings dengan HuggingFaceBgeEmbeddings
        # GROQ API tidak menyediakan embedding, jadi kita gunakan model embedding open-source
        self.embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        
        # Mengganti ChatGoogleGenerativeAI dengan ChatGroq
        # API key dilewatkan langsung sebagai argumen
        self.llm = ChatGroq(
            model_name="llama3-8b-8192",
            groq_api_key="<ganti dengan GROQ API Key anda>"
        )

        self.prompt = hub.pull("rlm/rag-prompt")
        self.vector_store = None
        
    def load_vector_store(self):
        self.vector_store = FAISS.load_local(
            "faiss_index", self.embeddings, allow_dangerous_deserialization=True
        )

    def retrieve(self, state: State):
        if self.vector_store is None:
            self.load_vector_store()
        retrieved_docs = self.vector_store.similarity_search(state["question"])
        return {"context": retrieved_docs}

    def generate(self, state: State):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = self.prompt.invoke({"question": state["question"], "context": docs_content})
        response = self.llm.invoke(messages)
        return {"answer": response.content}

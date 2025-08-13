import streamlit as st
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chat_models import init_chat_model
from langchain import hub
import os
import dotenv
import warnings

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

# Load environment variables
dotenv.load_dotenv('../pdf_cleaner/.env')

# # Initialize components
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
prompt = hub.pull("rlm/rag-prompt")

client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
vector_store = QdrantVectorStore(
    client=client,
    collection_name="chapter_chunks",
    embedding=embeddings,
)

llm = init_chat_model("gpt-4o-mini", model_provider="openai")

# # Define state
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# # Define steps
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

# # Compile graph
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

# # Streamlit UI
st.set_page_config(page_title="Bhagavad Gita QA Assistant", layout="wide")
st.title("📖 Bhagavad Gita QA Assistant")

question = st.text_input("Ask your question:", placeholder="e.g., What is karma according to Bhagavad Gita?")

if question:
    with st.spinner("Generating answer..."):
        response = graph.invoke({"question": question})
        retrieved_docs = response["context"]
        answer = response["answer"]

    st.subheader("📌 Answer")
    st.markdown(f"""
    <div style="background-color:#f0f8ff; padding:15px; border-radius:10px;">
        {answer}
    </div>
    """, unsafe_allow_html=True)

    st.subheader("📚 Retrieved Context")
    for i, doc in enumerate(retrieved_docs, 1):
        with st.expander(f"Document {i}"):
            st.markdown(doc.page_content)

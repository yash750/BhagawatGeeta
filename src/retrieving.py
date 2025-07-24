import warnings
import os

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

# Import necessary libraries
from langchain.prompts import PromptTemplate
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chat_models import init_chat_model
import dotenv


dotenv.load_dotenv('../pdf_cleaner/.env')

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

prompt = PromptTemplate(
    input_variables=["question", "context"],
    template="""
    You are a helpful assistant that answers questions about Bhagavad Gita.
    You are an assistant that only rephrases the input context. 
    Do not add, remove, or invent any new information. 
    Base your answer strictly on the context provided below.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """,
)

client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
vector_store = QdrantVectorStore(
    client=client,
    collection_name="chapter_chunks",
    embedding=embeddings,
)

# llm = init_chat_model("mistral-large-latest", model_provider="mistralai")
# llm = init_chat_model("gpt-4o-mini", model_provider="openai")
llm =  ChatOpenAI(temperature=0,openai_api_key=os.getenv("OPENROUTER_API_KEY"), openai_api_base=os.getenv("OPENAI_API_BASE"))

# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


# Define application steps
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

if __name__ == "__main__":
    # Compile application and test
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()

    response = graph.invoke({"question": "Can you explain best verses of chapter 3 from Bhagavad Gita?"})
    retrieved_docs = response["context"]

    print("---------------------#############-----------------------")
    print(f"✅ Total documents: {len(retrieved_docs)}")
    for doc in retrieved_docs:
        print(doc.page_content)
        print("---------------------#############-----------------------")
    
    print("---------------------#############-----------------------")
    print("Answer : ",response["answer"])


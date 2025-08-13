from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant
from langchain_core.documents import Document
import dotenv
import os
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings



dotenv.load_dotenv('../pdf_cleaner/.env')
df = pd.read_csv("../data/Bhagwad/Bhagwad_Gita.csv")

def make_docs(df):
    docs = []
    for _, row in df.iterrows():
        #take multiple columns as content
        content = row["Shloka"] + "\n" + row["Transliteration"] + "\n" + row["HinMeaning"] + "\n" + row["EngMeaning"] + "\n" + row["WordMeaning"]
        metadata = {
            "ID": row["ID"],
            "chapter": row["Chapter"],
            "verse": row["Verse"]
        }
        doc = Document(page_content=content, metadata=metadata)
        docs.append(doc)
    return docs

def store_embeddings(docs):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    # Don't manually create QdrantClient, let langchain do it
    qdrant = Qdrant.from_documents(
        documents=docs,
        embedding=embeddings,
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        collection_name="chapter_chunks",
    )

    # If you still want to return the raw client for inspection
    client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
    return client

if __name__ == "__main__":
    docs = make_docs(df)
    print(f"✅ Total documents: {len(docs)}")
    print(f"🧾 Sample document: {docs[0]}")
    print("---------------------#############-----------------------")

    qdrant_client = store_embeddings(docs)

    print("📦 Collection Info:")
    print(qdrant_client.get_collection(collection_name="chapter_chunks"))


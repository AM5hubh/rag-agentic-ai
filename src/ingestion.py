from langchain_community.document_loaders import PyPDFLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter 
# from langchain_openai import OpenAIEmbeddings 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

from src.config import PINECONE_INDEX_NAME

def run_ingestion(pdf_path: str, index_name: str):
    loader  = PyPDFLoader(pdf_path)
    documents = loader.load()

    print(f"Loaded {len(documents)} documents from {pdf_path}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=index_name,
    )

    print(f"Successfully ingested {len(chunks)} chunks into Pinecone index '{index_name}'")

    return vectorstore

if __name__ == "__main__":
    run_ingestion("data/Ebook-Agentic-AI.pdf", index_name=PINECONE_INDEX_NAME)
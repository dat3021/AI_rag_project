import os
import uuid
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from loader import load_markdown_from_github
from splitter import split_markdown_documents


def embedding():

    load_dotenv()

    # 1. Load documents from GitHub
    documents = load_markdown_from_github(repo_owner="dat3021", repo_name="RAG_document")
    
    if not documents:
        print("No documents found or failed to load. Exiting embedding process.")
        return None

    # 2. Split documents into chunks
    chunks = split_markdown_documents(documents)

    # 3. Setup Embedding Model
    gemini_embedding = GoogleGenerativeAIEmbeddings(
        google_api_key=os.environ.get("GOOGLE_API_KEY"),
        model="gemini-embedding-001",
    )
    
    # 4. Generate unique IDs for each chunk
    custom_ids = []
    for chunk in chunks:
         # Use the source file name and a UUID to ensure every chunk has a unique ID
         file_name = chunk.metadata.get("source", "doc")
         custom_ids.append(f"github_{file_name}_{uuid.uuid4().hex[:8]}")
        
    
    # 5. Store in ChromaDB
    db = Chroma.from_documents(
        documents=chunks,
        embedding=gemini_embedding,
        persist_directory="./vector_db",
        ids=custom_ids
    )
    
    # Optional test embed to check dimensions
    vector = gemini_embedding.embed_query(chunks[0].page_content)

    return db

if __name__ == "__main__":
    embedding = embedding()
    print(embedding.get())
    
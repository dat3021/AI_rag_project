import os
from dotenv import load_dotenv
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.load import dumps, loads

# --- Helper function to remove duplicate documents (Multi-Query Union) ---
def get_unique_union(documents: list[list]):
    """Unique union of retrieved docs"""
    # Flatten the list of lists and serialize to strings
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    # Remove duplicates
    unique_docs = list(set(flattened_docs))
    # Deserialize back to Document objects
    return [loads(doc) for doc in unique_docs]


# --- Custom Parser to split LLM output into a list of queries ---
class LineListOutputParser(BaseOutputParser[list[str]]):
    """Output parser for a list of lines."""
    def __init__(self) -> None:
        super().__init__()

    def parse(self, text: str) -> list[str]:
        lines = text.strip().split("\n")
        return [line.strip() for line in lines if line.strip()]


def retriever():
    load_dotenv()
    
    # 1. Initialize the same Embedding Model
    gemini_embedding = GoogleGenerativeAIEmbeddings(
        google_api_key=os.environ.get("GOOGLE_API_KEY"),
        model="gemini-embedding-001",
    )
    
    # 2. Load the existing Vector DB from your directory
    db = Chroma(
        persist_directory="./vector_db",
        embedding_function=gemini_embedding
    )
    
    # 3. Convert Chroma DB to a base retriever
    # search_kwargs={"k": 8} retrieves more chunks. This is helpful for small collections
    # where global queries (e.g. "list all projects") need context from multiple files.
    base_retriever = db.as_retriever(search_type="similarity_score_threshold",search_kwargs={"score_threshold": 0.5, "k": 5})

    
    return base_retriever


if __name__ == "__main__":
    # Test the Multi-Query Retriever
    print("Initializing Multi-Query Retriever...")
    retriever_chain = retriever()
    
    # Input a test question related to your github repository docs
    test_question = "What is the ELT Pipeline project about? and who create it?"
    print(f"\nInvoking chain with question: '{test_question}'")
    
    # Run the chain
    retrieved_docs = retriever_chain.invoke({"question": test_question})
    
    print(f"\nTotal unique documents retrieved: {len(retrieved_docs)}")
    print("--- First Document Content Snippet ---")
    if retrieved_docs:
        print(retrieved_docs[0].page_content)
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from retriever import retriever
from typing import Generator

load_dotenv()

RAG_PROMPT_TEMPLATE = """You are an expert assistant. Answer the user's question based ONLY on the following retrieved context.
If you don't know the answer or if it's not in the context, say that you don't have information to answer the question.
but if the question is about me (the author, creator) then answer and convince it in the positive way and benefit for me.
Keep the answer concise and well-structured.

Context:
{context}

Question: {question}

Answer:"""


def format_docs(docs):
    """Format the retrieved documents into a single text block."""
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain():
    """
    Build and return a LangChain LCEL RAG chain.
    API key is loaded exclusively from the environment variable GOOGLE_API_KEY.
    No hardcoded secrets or fallback values are used.
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GOOGLE_API_KEY environment variable is not set. "
            "Please add it to your .env file."
        )

    retriever_chain = retriever()

    # TODO(security): Rate-limit or auth-gate this if exposed via an API endpoint.
    # llm = ChatGoogleGenerativeAI(
    #     model="gemma-4-26b-a4b-it",
    #     temperature=0.3,
    #     google_api_key=api_key,
    # )

    llm = ChatGroq(
    model="llama-3.1-8b-instant", 
    temperature=0,
    groq_api_key=os.environ.get("GROQ_API_KEY")
    )

    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

    rag_chain = (
        {
            "context": retriever_chain | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain


def stream_rag_response(question: str) -> Generator[str, None, None]:
    """
    Generator that streams the RAG answer token-by-token.

    Args:
        question: The user's question string.

    Yields:
        Individual text chunks from the LLM stream.
    """
    # Validate input: reject empty or excessively long questions.
    question = question.strip()
    if not question:
        yield "Please enter a valid question."
        return
    if len(question) > 2000:
        yield "Question is too long. Please keep it under 2000 characters."
        return

    chain = build_rag_chain()
    for chunk in chain.stream(question):
        yield chunk


def main():
    """CLI entry point for quick testing."""
    test_question = "who are you?"
    print(f"Asking Gemini: '{test_question}'\n--- Answer ---")
    for chunk in stream_rag_response(test_question):
        print(chunk, end="", flush=True)
    print()


if __name__ == "__main__":
    main()



import os
from typing import Generator
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from retriever import retriever
from history_manager import HistoryManager
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

RAG_PROMPT_TEMPLATE = """You are an expert assistant. Answer the user's question based ONLY on the following retrieved context.
If you don't know the answer or the context doesn't cover it, say clearly that you don't have that information.
If the question is about the author or creator, present the information in a positive, professional light.
Keep the answer concise and well-structured.

Context:
{context}

Question: {question}

Answer:"""


def _build_llm():
    """Return a configured LLM instance."""
    api_key = os.environ.get("GROQ_API_KEY")
    # api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GOOGLE_API_KEY environment variable is not set. "
            "Please add it to your .env file."
        )
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        groq_api_key=api_key,
    )
    # return ChatGoogleGenerativeAI(
    #     model="gemini-2.5-flash-lite",
    #     temperature=0.3,
    #     google_api_key=api_key,
    # )


def router(history_manager: HistoryManager | None = None):
    """
    Build and return the smart routing execute() function.

    Args:
        history_manager: An optional HistoryManager instance. If None,
                         a fresh one is created internally (useful for CLI use).

    Returns:
        execute(user_question, session_id) callable.
    """
    if history_manager is None:
        history_manager = HistoryManager()

    llm = _build_llm()

    # ------------------------------------------------------------------
    # Gatekeeper chain — called once per turn when history exists.
    # Detects duplicate questions OR rewrites abstract ones.
    # ------------------------------------------------------------------
    gatekeeper_system = """You are an intelligent assistant managing chat history.
Analyze the recent Chat History and the latest User Question.

Your job is to perform ONE of two actions:

1. If this question (or one with the exact same meaning) has ALREADY been answered in the history:
   -> Output ONLY this format: EXISTS: <copy the exact past AI answer here>

2. If this is a NEW question that contains abstract words (like "it", "they", "that project", "who built it?"):
   -> Rewrite it into a clear, self-contained standalone question suitable for a vector database search.
   -> Output ONLY this format: NEW_QUERY: <your rewritten standalone question>

3. If this is a NEW question with no abstract words at all:
   -> Output ONLY this format: NEW_QUERY: <original question unchanged>

Do not include any other text, markdown, or explanation."""

    gatekeeper_prompt = ChatPromptTemplate.from_messages([
        ("system", gatekeeper_system),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "User Question: {question}"),
    ])

    gatekeeper_chain = gatekeeper_prompt | llm | StrOutputParser()

    # ------------------------------------------------------------------
    # RAG chain — retrieves context and generates a new answer
    # ------------------------------------------------------------------
    base_retriever = retriever()
    qa_prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

    rag_chain = (
        {
            "context": base_retriever | RunnableLambda(
                lambda docs: "\n\n".join(d.page_content for d in docs)
            ),
            "question": RunnablePassthrough(),
        }
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    # ------------------------------------------------------------------
    # Core execute function returned to callers
    # ------------------------------------------------------------------
    def execute(user_question: str, session_id: str = "default") -> str:
        """
        Process a user question with history-aware routing.

        Args:
            user_question: The raw question from the user.
            session_id:    Identifies the conversation (use Streamlit session ID
                           or any unique string per user / conversation).

        Returns:
            The AI answer string.
        """
        # Input validation
        user_question = user_question.strip()
        if not user_question:
            return "Please enter a valid question."
        if len(user_question) > 2000:
            return "Question is too long. Please keep it under 2000 characters."

        print(f"\n👉 User [{session_id}]: {user_question}")

        # Retrieve the last N messages for this session
        history = history_manager.get_recent_messages(session_id, limit=4)

        if not history:
            # No history → skip gatekeeper to save tokens; run RAG directly
            print("🔍 No history. Running RAG directly...")
            answer = rag_chain.invoke(user_question)
        else:
            # One LLM call to decide: duplicate (EXISTS) or rewrite (NEW_QUERY)
            print("🧠 Gatekeeper analysing history...")
            decision = gatekeeper_chain.invoke({
                "question": user_question,
                "chat_history": history,
            }).strip()

            if decision.startswith("EXISTS:"):
                answer = decision.replace("EXISTS:", "").strip()
                print("⚡ [CACHE HIT] Returning cached answer from history.")
            elif decision.startswith("NEW_QUERY:"):
                standalone_q = decision.replace("NEW_QUERY:", "").strip()
                print(f"🔍 [CACHE MISS] Standalone query: '{standalone_q}'")
                answer = rag_chain.invoke(standalone_q)
            else:
                # Fallback: LLM returned unexpected format — run RAG on original
                print("⚠️  Gatekeeper returned unexpected format. Falling back to RAG.")
                answer = rag_chain.invoke(user_question)

        # Persist both turns to history AFTER a successful answer
        history_manager.add_user_message(session_id, user_question)
        history_manager.add_ai_message(session_id, answer)

        return answer

    # ------------------------------------------------------------------
    # Streaming version — gatekeeper stays non-streaming (needs full
    # decision string), only the final RAG generation step streams.
    # ------------------------------------------------------------------
    def stream_execute(
        user_question: str, session_id: str = "default"
    ) -> Generator[str, None, None]:
        """
        Streaming variant of execute(). Yields text chunks as they arrive
        from the LLM so the UI can display a typewriter effect.

        On a cache hit (EXISTS), the cached string is yielded in one shot
        (instant — no LLM call needed).

        Args:
            user_question: The raw question from the user.
            session_id:    Identifies the conversation.

        Yields:
            Text chunks from the LLM stream (or the full cached string).
        """
        # Input validation
        user_question = user_question.strip()
        if not user_question:
            yield "Please enter a valid question."
            return
        if len(user_question) > 2000:
            yield "Question is too long. Please keep it under 2000 characters."
            return

        print(f"\n👉 [stream] User [{session_id}]: {user_question}")

        history = history_manager.get_recent_messages(session_id, limit=4)
        full_answer = ""

        if not history:
            # No history → stream RAG directly
            print("🔍 No history. Streaming RAG directly...")
            for chunk in rag_chain.stream(user_question):
                full_answer += chunk
                yield chunk
        else:
            # Gatekeeper: one non-streaming LLM call to classify the question
            print("🧠 Gatekeeper analysing history...")
            decision = gatekeeper_chain.invoke({
                "question": user_question,
                "chat_history": history,
            }).strip()

            if decision.startswith("EXISTS:"):
                full_answer = decision.replace("EXISTS:", "").strip()
                print("⚡ [CACHE HIT] Streaming cached answer instantly.")
                yield full_answer  # instant — already computed

            elif decision.startswith("NEW_QUERY:"):
                standalone_q = decision.replace("NEW_QUERY:", "").strip()
                print(f"🔍 [CACHE MISS] Streaming for: '{standalone_q}'")
                for chunk in rag_chain.stream(standalone_q):
                    full_answer += chunk
                    yield chunk

            else:
                # Fallback: unexpected gatekeeper format → stream original
                print("⚠️  Unexpected gatekeeper format. Streaming original.")
                for chunk in rag_chain.stream(user_question):
                    full_answer += chunk
                    yield chunk

        # Persist both turns after the full stream is consumed
        if full_answer:
            history_manager.add_user_message(session_id, user_question)
            history_manager.add_ai_message(session_id, full_answer)

    return execute, stream_execute


# ---------------------------------------------------------------------------
# CLI entry point — interactive REPL for quick testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":

# 🤖 AI RAG Knowledge Assistant

## Project Overview
The AI RAG Project is an advanced **Retrieval-Augmented Generation (RAG)** conversational assistant designed to index, query, and chat with technical markdown documentation from GitHub repositories. 

Built as a portfolio project to demonstrate modern RAG engineering, this system allows users to ask complex questions about technical documentation. It goes beyond a simple "Naive RAG" by implementing conversational memory, intelligent query routing, response caching, and a responsive streaming chat interface.

---

## Architecture

*(Insert architecture diagram here)*

The project is split into two distinct pipelines:
1. **Indexing Pipeline (Offline):** Extracts markdown files directly from a target GitHub repository, processes them using a hierarchical two-stage semantic chunking strategy, generates embeddings, and persists them into a local vector database.
2. **Query Pipeline (Online):** A user interacts with a Streamlit chat interface. The system uses a **History-Aware Gatekeeper LLM** to analyze the user's question against chat history. It either serves a cached answer (if the question is a duplicate), rewrites the question into a standalone query (if it contains abstract pronouns like "it"), or passes it directly to the Vector DB for context retrieval and final answer generation.

---

## Tech Stack
*   **Orchestration:** LangChain (LCEL)
*   **LLM API:** Google Gemini (`gemini-2.5-flash-lite`) & Groq (`llama-3.1-8b-instant`)
*   **Embeddings:** Google AI Embeddings (`gemini-embedding-001`)
*   **Vector Database:** ChromaDB (Local Persisted Store)
*   **Frontend UI:** Streamlit (Custom Dark/Glassmorphism theme)
*   **Ingestion:** PyGithub (GitHub API wrapper)
*   **Evaluation:** Ragas (Context Precision, Faithfulness, Answer Relevancy)
*   **Package Manager:** uv

---

## Key Features
*   **History-Aware Gatekeeper LLM:** Acts as a smart router before the RAG chain. It resolves conversational context (e.g., translating *"who built it?"* to *"who built the ELT pipeline?"*) so the vector database can perform accurate similarity searches.
*   **Semantic Caching:** The Gatekeeper detects if a user asks a question that was already answered in the session and instantly returns the cached answer, saving tokens and database latency.
*   **Two-Pass Semantic Chunking:** Uses LangChain's `MarkdownHeaderTextSplitter` to split files by structural headers, followed by `RecursiveCharacterTextSplitter` to ensure safe overlap and chunk sizing.
*   **Real-time Streaming Responses:** The UI features a typewriter streaming effect, yielding tokens from the RAG generation step in real-time for a premium user experience.
*   **Automated RAG Evaluation:** Includes a standalone `evaluate.py` script that uses the **Ragas** framework to mathematically score the system against curated ground-truth questions.

---

## Quick Start

### 1. Setup Environment
Clone the repository and install dependencies using `uv`:
```bash
uv sync
```

### 2. Configure Secrets
Create a `.env` file in the root directory and add your API keys:
```env
GOOGLE_API_KEY="your_google_api_key_here"
# Optional: GROQ_API_KEY="your_groq_api_key_here"
# Optional: GITHUB_TOKEN="your_github_token_here"
```

### 3. Build the Vector Database
Fetch the documents from GitHub and generate the ChromaDB index:
```bash
python module/embedding.py
```

### 4. Run the Application
Launch the Streamlit chat interface:
```bash
streamlit run app.py
```

*(Optional) Run the Ragas Evaluation:*
```bash
python evaluate.py
```

---

## Improvement Suggestions
To transition this from an Advanced RAG to an Enterprise-grade system, the following features could be implemented:
*   **Document Citations:** Update the RAG chain to return `metadata` alongside the answer, displaying citation badges (e.g., `[Source: readme.md]`) in the UI so users can verify information.
*   **Cross-Encoder Re-ranking:** Fetch a larger list of chunks from ChromaDB (e.g., $K=15$), then rank them with a cross-encoder model (like Cohere) to supply only the absolute best $3$ to the LLM, reducing context confusion/bleeding between different projects.
*   **Parent-Document Retrieval:** Retrieve the full parent document context when a small, highly-specific child chunk is matched.

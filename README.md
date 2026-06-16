# RAG-based LLM Customer Support Agent

An AI-powered customer support agent built with LangChain and Google Gemini 2.5 Flash, designed for businesses operating in the tax and accounting domain. The agent combines Retrieval-Augmented Generation (RAG) with autonomous tool calling to answer domain-specific queries, retrieve personal customer data, and escalate complex cases to human consultants.

## Architecture

The system is built on three main components:

- **LLM**: Google Gemini 2.5 Flash via LangChain, with windowed conversation memory (last 4 exchanges).
- **Vector Store**: ChromaDB for semantic retrieval over knowledge base documents.
- **Backend**: FastAPI server exposing the agent via REST API.
- **Frontend**: HTML/CSS/JS demo interface.

### RAG Pipeline

Knowledge base documents (`.docx`) are loaded, split into chunks (size: 600, overlap: 80), and embedded using `gemini-embedding-001`. At query time, the top-4 most relevant chunks are retrieved and injected into the prompt as context.

### Tool Calling

The agent autonomously decides which tool to invoke based on the user query:

| Tool | Trigger | Description |
|------|---------|-------------|
| `consulta_normativa_fiscale` | Tax or platform questions | RAG retrieval over the knowledge base |
| `consulta_dati_cliente` | Personal account questions | Lookup on customer Excel database |
| `escalation_customer_success` | Complex or out-of-scope queries | Flags the case for human follow-up |

Each response includes metadata on which tool was used (`RAG`, `CUSTOMER`,
`ESCALATION`, `GENERAL`, `RATE_LIMIT`), exposed both via API and in the frontend UI.

### Error Handling

- **Rate limiting**: detects Gemini 429 errors and returns a user-friendly retry message.
- **Empty responses**: automatic retry with 3s delay; falls back to escalation if still empty.
- **Thread safety**: agent state protected by a threading lock on the FastAPI side.

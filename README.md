# Smart IPO Analyzer

Smart IPO Analyzer is a Streamlit application that combines a LangGraph workflow, Tavily web search, and PDF-based retrieval to generate an IPO analysis report.

## Repository Analysis

- `app.py`: Streamlit UI, file upload flow, and graph execution.
- `ingestion.py`: loads PDFs from `data/`, splits them into chunks, creates OpenAI embeddings, and stores them in MongoDB.
- `src/agents.py`: defines the Supervisor, Market Scraper, RAG Analyst, and Risk Auditor agents.
- `src/graph.py`: wires the LangGraph state machine and routes between agents.
- `src/tools.py`: exposes the Tavily web-search tool and the MongoDB vector-search tool.

## Important Constraints

- This repository is a single web application service.
- It depends on external services for OpenAI, Tavily, and MongoDB.
- The PDF search flow uses MongoDB Atlas Vector Search with the `$vectorSearch` stage.
- Because of that Atlas dependency, the compose file does not include a local MongoDB container.
- Uploaded PDF files are stored in `./data` on the host and are mounted into the container.

## Prerequisites

- Docker Desktop or Docker Engine with Compose support
- An OpenAI API key
- A Tavily API key
- A MongoDB Atlas URI
- A MongoDB Atlas Vector Search index named `vector_index` on the `Smart_IPO.embeddings` collection

## Setup

1. Create a local env file.

```powershell
Copy-Item .env.example .env
```

2. Fill in your real secrets inside `.env`.

- Use `MONGO_URI=...` as the Mongo variable name.
- Keep the file in strict `KEY=VALUE` format because Docker Compose parses it before startup.
- Do not keep secrets in source control.

3. Start the application.

```bash
docker compose up --build
```

4. Open the app in your browser at `http://localhost:8501`.

## Ingesting PDFs

- Upload a PDF in the Streamlit sidebar and click `Process Document (Ingest)`.
- Or run ingestion manually from Docker:

```bash
docker compose run --rm app python ingestion.py
```

## Sharing This Project

- Share the repository without `.env`.
- Keep `.env.example` in the repo so other people know which variables they need.
- Tell recipients to create their own `.env` file and then run `docker compose up --build`.

## Troubleshooting

- If Docker Compose cannot start, confirm that `.env` exists next to `docker-compose.yml` and every line uses `KEY=VALUE` format.
- If PDF search fails, verify your Atlas collection is `Smart_IPO.embeddings` and the index name is `vector_index`.
- If MongoDB is reported as missing, confirm you used `MONGO_URI`, not a malformed custom name.

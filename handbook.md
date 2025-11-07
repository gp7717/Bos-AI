# Bos-AI Handbook

## 1. Project Overview
- **Purpose**: Provide a LangGraph-powered orchestrator that coordinates the SQL and computation agents to answer business analytics questions (e.g., “What were last month’s sales?”).
- **Key components**:
  - `Agents/orchestrator`: FastAPI service exposing `/query` and `/query/stream`.
  - `Agents/QueryAgent`: SQL agent built on LangGraph; handles schema lookup, query generation, validation, and execution.
  - `Agents/ComputationAgent`: Sandbox-based Python executor for light post-processing.
  - `Agents/QueryAgent/main.py`: Standalone CLI for running the SQL agent outside of the web server.

## 2. Environment Setup
1. **Python**: Use Python 3.10.
2. **Virtual environment** (PowerShell):
   ```powershell
   py -3.10 -m venv .venv
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
   .\.venv\Scripts\Activate.ps1
   ```
3. **Dependencies**:
   ```powershell
   python -m pip install --upgrade pip
   pip install -r Bos-AI/requirements.txt
   ```
4. **Environment variables** (example placeholders):
   ```powershell
   $env:AZURE_OPENAI_ENDPOINT="https://<resource>.cognitiveservices.azure.com/"
   $env:AZURE_OPENAI_API_KEY="<api-key>"
   $env:AZURE_OPENAI_DEPLOYMENT="gpt-5-mini"
   $env:DATABASE_URL="postgresql://user:password@host:port/dbname"
   # add any other secrets required by your deployment
   ```

## 3. Running the Orchestrator Server
```powershell
cd Bos-AI
uvicorn Agents.orchestrator.server:app --reload --host 0.0.0.0 --port 8000
```

### Endpoints
- **Health check**: `GET http://localhost:8000/health`
- **Synchronous query** (`POST /query`):
  ```json
  {
    "question": "What were last month's sales?",
    "trace": true,
    "include_data": true
  }
  ```
- **Streaming query** (`POST /query/stream`):
  - Returns newline-delimited JSON mirroring the standalone CLI stream.
  - Useful for debugging LangGraph transitions in real time.

All responses include `agent_results` and optional `trace` events showing planner decisions, tool invocations, and errors.

## 4. Standalone SQL Agent (CLI)
Run the SQL agent directly without the orchestrator:
```powershell
python -m Agents.QueryAgent.main "What were last month's sales?"
# Add --stream to show intermediate steps
```

The CLI is ideal when you want raw SQL output or need to debug the query path independently.

## 5. Logging & Diagnostics
- Structured logging through Python’s `logging` module.
- Streaming endpoint provides near real-time insight into LangGraph states.
- Increase verbosity with:
  ```powershell
  $env:LOG_LEVEL="DEBUG"
  ```
- Key log messages:
  - `Validation produced a trivial query; retaining original`
  - `Query executed successfully but returned no rows`
  - `Blocked unsafe SQL query (reason=...)`

## 6. SQL Agent Guardrails (Current State)
- Guardrails now rely solely on keyword blocking (`CREATE`, `DROP`, `ALTER`, `DELETE`, `TRUNCATE`, `EXEC`, `INVOKE`, etc.).
- Table whitelisting and CTE alias filtering have been removed; contextual restrictions come from the prompts (`table_context.yaml`).
- `_is_safe_query` will block only when those destructive keywords appear.
- The validator fallback retains the original SQL when the reviewer attempts to downgrade it (e.g., to `SELECT 1;`).

## 7. Runtime Notes
- **Zero-row results**: The agent now echoes the executed SQL so you can inspect filters/date ranges when a query returns no rows.
- **Legacy compatibility**: `_extract_table_tokens()` still exists as a no-op to support cached orchestrator imports.
- **Retry policy**: The orchestrator does not auto-retry SQL failures. Validation fallback keeps the original query, but the planner leaves error handling to the caller.

## 8. Sample Workflow
1. Export environment variables and start the server.
2. Submit a request (`/query` for synchronous results, `/query/stream` for live traces).
3. Inspect logs/stream output:
   - Planner chooses `sql` (computation optional unless explicitly needed).
   - SQL generation triggers guardrails; unsafe queries are blocked before execution.
4. If an error occurs:
   - Check `agent_results[].error` and the trace for context.
   - For SQL errors, rerun via the CLI to inspect the raw message or adjust prompts/schema context.
5. Confirm the final answer contains the expected numeric output; if not, review `last_query_result` and logging output.

## 9. Troubleshooting
- **Import errors** (`CompiledStateGraph` missing): Ensure `langgraph>=0.2.0` and adjust imports to `typing.Any`.
- **Sandbox violations**: Update `SafeComputationSandbox` allowed nodes or reinforce the computation prompt (no `import` statements).
- **Azure OpenAI issues**: Verify endpoint, API key, and deployment names; check HTTP status codes in logs.
- **SQL validator fallback**: Confirm the query isn’t being rewritten to a trivial statement. If it is, inspect the log message and adjust prompts or safety logic.
- **Missing totals in final answer**: When the SQL agent succeeds but the answer is empty, inspect `agent_results[].tabular` or re-run via CLI for the raw output.
- **Database connectivity**: Ensure `DATABASE_URL` is reachable from the orchestrator host.

## 10. Useful References
- `table_context.yaml`: Descriptions and key columns for approved tables (Shopify orders, attribution tables, etc.).
- `Agents/orchestrator/service.py`: Graph compilation, streaming serialization, `_serialise_chunk` for JSON-safe output.
- `Agents/QueryAgent/nodes.py`: Core SQL agent logic (generate → check → run → summarize).
- `Agents/ComputationAgent/sandbox.py`: AST checker configuration if the computation agent needs new syntax.
- `Agents/QueryAgent/main.py`: CLI entry point for direct SQL agent invocation.

---

Last updated: 2025-11-08.


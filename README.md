  Bos-AI Multi-Agent Orchestration
  ================================

  This document explains the current Bos-AI multi-agent architecture, the execution flow, and the components that make up the production implementation.

  High-Level Request Flow
  -----------------------

  ```mermaid
  flowchart TB
      subgraph Client Layer
          A[Caller\n(API/UI/CLI)]
      end
      subgraph Orchestrator Runtime
          B[orchestrator.service.run / arun]
          C[LangGraph Orchestrator\n(workflow.compile_orchestrator)]
          D[Planner\nplanner.plan]
          E[Agent Execution Loop\nworkflow.execute_agent]
      F[Composer\ncomposer.compose]
      end
      subgraph Specialised Agents
          G[SQL Agent\nQueryAgent/sql_agent.py]
          H[Computation Agent\nComputationAgent/agent.py]
          I[API Docs Agent\nApiDocsAgent/agent.py]
      end
      subgraph Data Services
          I[Database / MCP Provider]
          J[Safe Python Sandbox]
      end

      A -->|AgentRequest| B
      B --> C
      C -->|PlannerDecision| D
      D -->|ordered agents| E
      E -->|invoke sql (auto-retries)| G
      E -->|invoke computation| H
      E -->|invoke api docs| I
      G -->|read-only SQL queries| I
      H -->|sandboxed code| J
      G -->|AgentResult| E
      H -->|AgentResult| E
      E -->|AgentResults| F
      F -->|OrchestratorResponse| B
      B -->|answer + trace| A
  ```

  Detailed Workflow
  ------------------

  ### 1. Request Entry & Graph Compilation

  **Entry Point (`service.py`)**
  - `run()` / `arun()` receives an `AgentRequest` containing:
    - `question`: User query string
    - `context`: Optional key-value pairs (e.g., `api_base_url`, `current_datetime_utc`)
    - `prefer_agents` / `disable_agents`: Explicit agent selection overrides
    - `trace`: Boolean flag to include execution traces
    - `include_data`: Boolean flag to include tabular results

  - **Graph Compilation** (`_get_compiled_graph()`):
    - First invocation compiles the LangGraph workflow and caches it globally
    - Subsequent requests reuse the cached graph for performance
    - Graph structure: `START → plan → [execute_agent]* → compose → END`

  ### 2. State Initialization & Temporal Context

  **Initial State** (`OrchestratorState`):
  ```python
  {
      "request": AgentRequest (with temporal context attached),
      "planner": None,
      "pending_agents": [],
      "agent_results": [],
      "trace": [],
      "response": None
  }
  ```

  **Temporal Context Attachment** (`_attach_temporal_context()`):
  - Automatically injects UTC datetime and date boundaries if missing:
    - `current_datetime_utc`: ISO format UTC timestamp
    - `current_date`: ISO date string (YYYY-MM-DD)
    - `current_date_start_hour`: "{date} 00"
    - `current_date_end_hour`: "{date} 23"
  - Used by agents for time-bounded queries and API calls

  ### 3. Planner Decision-Making (`plan_node`)

  **Two-Stage Agent Selection:**

  **Stage 1: Heuristic Keyword Matching** (`_heuristic_candidates()`):
  - Scores question against keyword sets:
    - **SQL Keywords**: `table`, `tables`, `list`, `show`, `top`, `per`, `group`, `average`, `sum`, `count`, `trend`, `breakdown`, `report`
    - **Computation Keywords**: `calculate`, `difference`, `ratio`, `project`, `simulate`, `compute`, `forecast`, `estimate`, `compare`, `what is`, `percent`
    - **API Docs Keywords**: `api`, `endpoint`, `route`, `http`, `/api/`, `get /`, `post /`, `put /`, `delete /`, `patch /`, `status code`, `request body`

  - **Selection Logic**:
    - If `api_score >= max(sql_score, comp_score)` and `api_score > 0`: add `api_docs`
    - If `sql_score >= comp_score` and `sql_score > 0`: add `sql`
    - If `comp_score >= sql_score` and `comp_score > 0`: add `computation`
    - **Default fallback**: If no keywords match, default to `["sql"]`

  **Stage 2: Preference Application** (`_apply_preferences()`):
  - Applies user preferences in order:
    1. Adds preferred agents (if not disabled) to front of list
    2. Filters out disabled agents
    3. Preserves heuristic ordering for remaining agents

  **Stage 3: LLM Refinement**:
  - Sends candidates to LLM with prompt:
    - System: "Use proposed agent list, optionally drop agents that add no value, keep order unless compelling reason"
    - User: Question, context keys, preferred agents, candidates, disabled agents
  - LLM returns: `agents` (ordered list), `rationale` (explanation), `confidence` (0.0-1.0)
  - **Fallback**: If LLM fails, uses heuristic candidates with confidence 0.4

  **Output** (`PlannerDecision`):
  - `chosen_agents`: Ordered tuple of agent names
  - `rationale`: LLM explanation of selection
  - `confidence`: Confidence score (0.0-1.0)
  - `guardrails`: Applied constraints (e.g., disabled agents)

  **Trace Event**: Records planner decision with agent list and confidence

  ### 4. Agent Execution Loop (`execute_agent`)

  **Sequential Execution**:
  - Processes agents one-by-one from `pending_agents` queue
  - Pops first agent, invokes appropriate handler, appends result

  **Agent Invocation**:

  **SQL Agent** (`_run_sql_agent()`):
  - **Retry Logic**: Up to 4 attempts (initial + 3 retries)
  - **Per-Attempt Flow**:
    1. Builds `SQLAgentState` with question and context
    2. Invokes compiled SQL LangGraph workflow
    3. Extracts query result from state
    4. Checks success via `result_payload.success`
    5. On failure: Builds retry feedback with error details and failing query
    6. Appends feedback as `HumanMessage` to message history
    7. Retries with enriched context
  - **Success**: Returns `AgentResult` with `TabularResult` (columns, rows, row_count) and natural language answer
  - **Failure**: Returns `AgentResult` with error details and attempt history
  - **Trace Events**: One per attempt (success/failure with query, row_count, latency)

  **API Docs Agent** (`_run_api_docs_agent()`):
  - **Planning Phase** (`_plan_api_call()`):
    1. Tokenizes question and scores against endpoint tokens
    2. Selects top-k relevant endpoints from `Docs/api_docs_context.yaml`
    3. Selects top-k relevant documentation snippets
    4. Formats candidates as numbered list with method, path, description
    5. Sends to LLM with question, candidates, and context
    6. LLM returns `ApiCallPlan`:
      - `make_request`: Boolean (should execute?)
      - `method`: HTTP verb
      - `path`: Relative path (e.g., `/api/procurement/products`)
      - `path_params`: Template substitutions (e.g., `:productId` → value)
      - `query_params`: Query string parameters
      - `json_body`: Request body for POST/PUT/PATCH
      - `headers`: Additional headers
      - `reason`: Justification
      - `failure_reason`: Explanation if `make_request=false`
  - **Force Attempt Logic** (`_should_force_attempt()`):
    - If planner declined but method is GET/POST and reason contains keywords like "documentation", "not specify", "cannot invent": forces attempt anyway
  - **Execution Phase** (`_execute_plan()`):
    1. **Base URL Resolution** (`_resolve_base_url()`):
      - Checks: `context["api_base_url"]` → `context["base_url"]` → `API_AGENT_BASE_URL` env → `DEFAULT_BASE_URL`
      - Default: `https://dashbackend-a3cbagbzg0hydhen.centralindia-01.azurewebsites.net`
    2. **Path Rendering** (`_render_path()`):
      - Substitutes `:param` and `{param}` with values from `path_params`
      - Validates all placeholders resolved
    3. **Header Building** (`_build_headers()`):
      - Adds `Accept: application/json`
      - Merges `context["api_headers"]` and plan headers
      - **Firebase Auth** (`_get_firebase_token()`):
        - Checks: direct token → cached token → token file → email/password generation
        - Adds `Authorization: Bearer {token}` if available
    4. **HTTP Request**:
      - Uses `httpx.Client` with timeout (default 30s)
      - Retries on 401 (token refresh) and 400 (auto-adjust params)
      - Captures response: status_code, body, JSON, elapsed time
    5. **Summarization** (`_summarise_execution()`):
      - Sends response to LLM with question, method, URL, status, body snippet
      - Returns natural language summary
  - **Trace Events**: Decision (planned call), Tool (executed request), Message (summarized)

  **Computation Agent**:
  - Invokes `ComputationAgent.invoke()` with question and context
  - Returns `AgentResult` with computation output or error

  **Result Accumulation**:
  - Appends `AgentResult` to `agent_results` list
  - Merges agent trace events into orchestrator trace
  - Updates state: `pending_agents` (decremented), `agent_results` (appended), `trace` (merged)

  **Routing** (`_route_after_execute()`):
  - If `pending_agents` non-empty: return to `execute_agent`
  - If empty: proceed to `compose`

  ### 5. Composer Aggregation (`compose_node`)

  **Input Processing**:
  - Extracts question, planner rationale, and all `AgentResult` objects
  - Builds agent summaries: `"Agent: {name} | status: {status}. Answer: {answer} Error: {error}"`

  **Tabular Data Selection** (`_select_tabular()`):
  - Returns first `TabularResult` from successful agents (if any)
  - Used for structured data responses

  **LLM Composition**:
  - **System Prompt**:
    - Role: Senior Business Intelligence Agent
    - Instructions:
      - Concise, direct answers (no explanations)
      - Omit metrics that are 0 or unavailable
      - Prefer structured metrics over narrative
      - Use Indian Rupees format: `Rs.1,75,206.00` (not ₹ or INR)
      - Professional tone, no emojis/hashtags
      - If no data: state "no data was returned"
  - **User Prompt**:
    - Question
    - Planner rationale
    - Agent summaries (one per line)
  - LLM generates final consolidated answer

  **Response Construction** (`OrchestratorResponse`):
  - `answer`: LLM-generated consolidated answer
  - `data`: Selected `TabularResult` (if any)
  - `agent_results`: All agent results (for debugging/transparency)
  - `metadata`: Planner confidence, agent list
  - `trace`: Full execution trace (if `request.trace=true`)

  **Trace Event**: Records composer completion

  ### 6. Response Return

  **Final State**:
  - `response`: `OrchestratorResponse` with answer, data, traces
  - `trace`: Complete execution timeline
  - `request`: Original request (with temporal context)

  **Return Path**:
  - `compose_node` → `END`
  - `service.py` extracts `response` from final state
  - Returns to caller (API/UI/CLI)

  Component Breakdown
  -------------------

  - **Entry points (`Agents/orchestrator/service.py`, `server.py`, `main.py`)**
    - `run`/`arun` expose synchronous and asynchronous orchestrator execution with automatic graph compilation and logging setup.
    - `server.py` wires the orchestrator into the API surface (FastAPI/Starlette) for deployment.

  - **Orchestrator workflow (`Agents/orchestrator/workflow.py`)**
    - Builds a `StateGraph` with `plan → execute_agent → compose`.
    - Persists `OrchestratorState`: request payload, planner decision, pending queue, accumulated `AgentResult` objects, and execution `TraceEvent`s.
    - Sequentially invokes agents chosen by the planner and records per-agent latency, output, and errors.

  - **Planner (`Agents/orchestrator/planner.py`)**
    - Combines heuristic keyword matching with an LLM (`AzureChatOpenAI`) to rank eligible agents.
    - Produces a `PlannerDecision` containing rationale, agent ordering, confidence score, and applied guardrails.
    - Applies preference/disable lists from the incoming `AgentRequest`.

  - **Composer (`Agents/orchestrator/composer.py`)**
    - Uses an LLM prompt to consolidate agent answers into the final `OrchestratorResponse`.
    - Selects a single `TabularResult` to return (first successful agent with tabular data).
    - Annotates metadata with planner confidence and agent participation.

  - **SQL Agent (`Agents/QueryAgent`)**
    - Implemented as a LangGraph workflow (`sql_agent.compile_sql_agent`), exposing nodes defined in `nodes.py`.
    - Key stages: table overview, optional schema fetch, query generation, query checking, tool execution via MCP client, and result interpretation.
    - Automatically retries failed SQL executions (default 3 attempts). Each retry feeds the database error back into the LLM so follow-up queries can self-correct.
    - Enforces read-only usage through forbidden pattern checks and explicit allowed-table filters.

  - **API Docs Agent (`Agents/ApiDocsAgent`)**
    - Loads curated context from `Docs/api_docs_context.yaml`, retrieves the most relevant snippets, and answers REST API questions with the shared Azure LLM.
    - **Base URL Resolution**: Checks context → environment variables → hardcoded default
    - **Endpoint Selection**: Token-based scoring matches question to documented endpoints
    - **Planning**: LLM selects endpoint and constructs HTTP request plan
    - **Execution**: Makes HTTP requests with Firebase auth, handles 401/400 retries
    - Returns rich trace events referencing the documentation chunks used, enabling auditable responses.

  - **Computation Agent (`Agents/ComputationAgent/agent.py`)**
    - Generates a computation plan and safe Python payload with LLM prompts and parses them into `_ComputationPlan`.
    - Executes code inside `SafeComputationSandbox`, capturing stdout, locals, and results; violations raise `SandboxViolation`.
    - Summarises outputs into `_ComputationSummary`, returning natural-language answers and optional structured tables.

  - **Shared models and tracing (`Agents/core/models.py`)**
    - Defines `AgentRequest`, `AgentResult`, `TabularResult`, `TraceEvent`, and orchestration enums.
    - Centralises type coercion, error envelopes, and trace metadata (timestamps, event types).

  State Transition Diagram
  -------------------------

  ```mermaid
  stateDiagram-v2
      [*] --> Entry: AgentRequest received
      Entry --> Plan: Initialize OrchestratorState
      Plan --> Plan: Attach temporal context
      Plan --> Plan: Heuristic keyword matching
      Plan --> Plan: Apply preferences
      Plan --> Plan: LLM refinement
      Plan --> ExecuteAgent: Agents selected
      Plan --> Compose: No agents selected
      ExecuteAgent --> ExecuteAgent: Process next agent
      ExecuteAgent --> SQLAgent: agent == 'sql'
      ExecuteAgent --> ApiDocsAgent: agent == 'api_docs'
      ExecuteAgent --> ComputationAgent: agent == 'computation'
      SQLAgent --> SQLAgent: Retry on failure (max 4 attempts)
      ApiDocsAgent --> ApiDocsAgent: Retry on 401/400
      ExecuteAgent --> ExecuteAgent: More agents pending
      ExecuteAgent --> Compose: All agents complete
      Compose --> Compose: Select tabular data
      Compose --> Compose: LLM composition
      Compose --> [*]: OrchestratorResponse
  ```

  SQL Agent Detailed Flow
  -----------------------

  ```mermaid
  flowchart TB
      A[User Question\nSQLAgentState.messages] --> B[list_tables node]
      B --> C{Schema needed?}
      C -->|Yes| D[call_get_schema node\nMCP tool: mcp_db_schema]
      C -->|No| E[generate_query node]
      D --> E
      E --> F[LLM generates SQL query\nwith table context]
      F --> G[safety_check node]
      G -->|Forbidden keywords| H[Blocked\nReturn error message]
      G -->|Unauthorized tables| H
      G -->|Valid| I[check_query node]
      I --> J[LLM reviews query\nfor correctness]
      J --> K[execute_query node]
      K --> L[MCP tool: mcp_db_query]
      L --> M{Success?}
      M -->|Yes| N[parse_results node]
      M -->|No| O[Extract error message]
      O --> P{Retries left?}
      P -->|Yes| Q[Build retry feedback\nwith error + query]
      Q --> E
      P -->|No| R[Return failure\nwith attempt history]
      N --> S[Attach TabularResult\ncolumns, rows, row_count]
      S --> T[Return AgentResult\nto orchestrator]
      H --> T
      R --> T
  ```

  **SQL Agent Retry Logic**:
  - **Maximum Attempts**: 4 (initial + 3 retries)
  - **Retry Trigger**: Query execution failure (database error, syntax error, etc.)
  - **Retry Feedback Format**:
    ```
    The previous SQL query failed to execute.
    Error details: {error_text}
    Failing query:
    {query}
    Generate a corrected read-only SQL query that addresses the error and try again using only the authorised tables.
    ```
  - **State Preservation**: Each retry maintains message history, allowing LLM to learn from previous attempts
  - **Success Criteria**: `result_payload.success == True` with valid data
  - **Failure Criteria**: All retries exhausted or configuration error

  **Safety Checks** (`safety_check`):
  - **Forbidden Keywords**: `CREATE`, `DROP`, `ALTER`, `TRUNCATE`, `INSERT`, `UPDATE`, `DELETE`, `MERGE`, `GRANT`, `REVOKE`, `CALL`, `EXEC`, `EXECUTE`, `INVOKE`
  - **Table Whitelist**: Only queries referencing tables in `allowed_tables` configuration
  - **Pattern Matching**: Case-insensitive regex matching against forbidden keywords
  - **Error Response**: Returns blocked message explaining why query was rejected

  API Docs Agent Detailed Flow
  -----------------------------

  ```mermaid
  flowchart TB
      A[User Question] --> B[Tokenize question]
      B --> C[Score endpoints\nfrom api_docs_context.yaml]
      C --> D[Select top-k endpoints]
      D --> E[Select top-k doc snippets]
      E --> F[Format candidates\nmethod, path, description]
      F --> G[LLM Planning\nApiCallPlan generation]
      G --> H{make_request?}
      H -->|false| I{Force attempt?}
      I -->|Yes| J[Override: make_request=true]
      I -->|No| K[Return PlanningDeclined error]
      H -->|true| L[Resolve base URL]
      J --> L
      L --> M[Render path\nsubstitute :param]
      M --> N[Build headers\n+ Firebase auth]
      N --> O[Execute HTTP request]
      O --> P{Status code?}
      P -->|401| Q[Refresh Firebase token]
      Q --> O
      P -->|400| R[Auto-adjust params\nstartDateTime/endDateTime]
      R --> O
      P -->|2xx| S[Extract response\nbody, JSON, elapsed]
      P -->|Other| T[Capture error]
      S --> U[LLM Summarization]
      T --> V[Return error result]
      U --> W[Return success result\nwith summary]
      K --> V
  ```

  **API Docs Agent Planning**:
  - **Endpoint Selection**: Token-based scoring matches question tokens to endpoint tokens (method, path, description)
  - **Documentation Context**: Selects relevant documentation snippets for additional context
  - **LLM Planning**: LLM receives formatted candidate list and selects appropriate endpoint with full request specification
  - **Force Attempt**: If planner declines but reason suggests missing documentation (e.g., "not specify", "cannot invent"), forces attempt for GET/POST requests

  **Base URL Resolution Priority**:
  1. `context["api_base_url"]`
  2. `context["base_url"]`
  3. `API_AGENT_BASE_URL` environment variable
  4. `DEFAULT_BASE_URL` (hardcoded: `https://dashbackend-a3cbagbzg0hydhen.centralindia-01.azurewebsites.net`)

  **Firebase Authentication Flow**:
  1. Check direct token in context (`api_bearer_token`, `firebase_id_token`)
  2. Check cached token (if valid and not expired)
  3. Check token file (`FIREBASE_TOKEN_FILE` env or `api_token_file` context)
  4. Generate via email/password (`FIREBASE_WEB_API_KEY`, `FIREBASE_EMAIL`, `FIREBASE_PASSWORD`)
  5. Cache token with expiry (minus 120s grace period)

  **HTTP Retry Logic**:
  - **401 Unauthorized**: Attempts Firebase token refresh once, retries request
  - **400 Bad Request**: Auto-adjusts missing `startDateTime`/`endDateTime` query params from context, retries
  - **Other Errors**: Returns error result with status code and body snippet

  Error Handling & Resilience
  ----------------------------

  **Orchestrator-Level Error Handling**:
  - **Graph Compilation Failure**: Raises `RuntimeError` with details, prevents service startup
  - **Planner Failure**: Falls back to heuristic candidates with confidence 0.4
  - **Agent Execution Failure**: Captures error in `AgentResult`, continues to next agent
  - **Composer Failure**: Returns partial response with available agent results

  **Agent-Level Error Handling**:
  - **SQL Agent**: Retries up to 4 times with error feedback, returns failure after exhaustion
  - **API Docs Agent**: Handles HTTP errors (401/400 retries), planning failures return `PlanningDeclined`
  - **Computation Agent**: Sandbox violations raise `SandboxViolation`, captured as `AgentError`

  **Error Propagation**:
  - Errors captured in `AgentError` with `message`, `type`, and `details`
  - Trace events record errors with `TraceEventType.ERROR`
  - Failed agents still contribute to composer (with error context)
  - Final response includes all agent results (success and failure) for transparency

  Operational Considerations
  --------------------------

  - **Lifecycle & caching**
    - `_get_compiled_graph` caches the compiled LangGraph to avoid repeated graph builds per request.
    - Planner and agents lazily acquire shared resources through `Agents.QueryAgent.config.get_resources()`, centralising LLM credentials and MCP clients.

  - **Tracing & observability**
    - Each agent appends `TraceEvent` entries describing decisions, tool invocations, and errors.
    - The orchestrator merges agent traces, enabling downstream monitoring or replay through `OrchestratorResponse.trace`.

  - **Safety controls**
    - SQL agent: whitelist tables, block DDL/DML keywords, cross-check tool calls before execution.
    - Computation agent: sandbox restricts imports, system prompts forbid unsafe operations; violations surface as structured `AgentError`s.

  - **Configuration knobs**
    - `AgentRequest` exposes `prefer_agents`, `disable_agents`, `context`, `max_turns`, and tracing flags.
    - `Agents/core/settings.py` (not shown) houses environment-driven settings such as LLM endpoints and sandbox timeouts.

  Extensibility Guidelines
  ------------------------

  1. **Adding a new specialist agent**
    - Implement an `AgentResult`-producing client with trace logging similar to the existing agents.
    - Register the agent in `Agents/core/models.AgentName` and update planner heuristics plus composer summaries.
    - Extend the orchestrator’s execution switch in `workflow.execute_agent`.

  2. **Planner evolution**
    - Introduce capability metadata (e.g., required context keys, latency budgets) to drive smarter ordering.
    - Consider iterative planning: re-run the planner based on partial results when confidence < threshold.

  3. **Observability & testing**
    - Capture token usage and tool metrics inside `TraceEvent.data` for SLO dashboards.
    - Build integration tests with mocked LLM responses to validate planner choices, SQL safety gates, and sandbox failure handling.

  Reference: Multi-Agent Example
  ------------------------------

  For comparison, `Examples/multi-agent-collaboration.ipynb` demonstrates a conversational multi-agent network using LangGraph. Our production flow adopts a planner-driven pipeline rather than round-robin dialogue, but the notebook is a useful reference for alternative coordination strategies and tooling patterns.



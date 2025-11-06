# Agentic Assistant - Analytics Q&A System

A production-ready agentic assistant system for analytics queries across multiple data sources (Amazon Ads, Google Ads, Meta Ads, Shopify Sales, etc.).

## Architecture

The system follows a modular architecture using **LangChain** and **LangGraph**:

- **API Gateway**: Single entry point with auth, rate limiting, and request tracking
- **LangGraph Orchestrator**: State-based orchestration using LangGraph
- **Modular Agents**: Plug-and-play agents built with LangChain
  - **Router Agent V2**: Intent classification and slot extraction
  - **Planner Agent V2**: Execution plan generation
  - **Composer Agent V2**: Natural language answer generation
- **LangChain Tools**: Reusable tools for data access and computation
- **Data Access Agents**: Domain-specific agents (Ads, Sales, Amazon)
- **Guardrail Agent**: Validation and safety checks
- **Computation Agent**: Metric calculations and aggregations

See [MODULAR_ARCHITECTURE.md](docs/MODULAR_ARCHITECTURE.md) for details on the new architecture.

## Shared Services

- **Tool Registry**: Declarative catalog of tools and capabilities
- **Schema Registry**: Database schema metadata
- **Metric Dictionary**: Authoritative metric definitions
- **Memory Store**: Vector DB for conversation history
- **Observability**: Tracing, logging, and metrics
- **Policy Layer**: RBAC, PII masking, row-level security

## Quick Start

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Set up environment**:
```bash
cp .env.example .env
# Edit .env with your configuration
```

3. **Initialize database**:
```bash
psql -U postgres -f schema.sql
```

4. **Run the service**:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload


   cd "/Users/gauravpradhan/Desktop/SpacePeppers/Agentic /Assistant"
   source venv/bin/activate
   ./venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

5. **Test the API**:
```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{"query": "What was ROAS last week for SB campaigns in Delhi?"}'
```

## Project Structure

```
.
├── app/
│   ├── agents/          # Legacy agent implementations (still in use)
│   ├── agents_v2/       # LangChain-based agents (router, planner, composer)
│   ├── core/            # Base classes and agent registry
│   ├── tools/           # LangChain tools for data access and computation
│   ├── graph/           # LangGraph state graph for orchestration
│   ├── services/        # Shared services (orchestrator_v2, etc.)
│   ├── models/          # Pydantic models
│   ├── config/          # Configuration
│   ├── api/             # FastAPI routes
│   └── main.py          # Application entry point
├── config/              # YAML configuration files
├── docs/                # Documentation (including MODULAR_ARCHITECTURE.md)
├── tests/               # Test suite
├── schema.sql           # Database schema
└── requirements.txt     # Python dependencies
```

## Configuration

Key configuration files:
- `config/tools.yaml`: Tool Registry definitions
- `config/metrics.yaml`: Metric Dictionary
- `config/schemas.yaml`: Schema Registry

## Development

Run tests:
```bash
pytest tests/
```

Format code:
```bash
black app/
ruff check app/
```

## License

Proprietary - SpacePeppers


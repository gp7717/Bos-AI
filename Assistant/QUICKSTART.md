# Quick Start Guide

## Setup

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Configure environment:**
```bash
# Create .env file with required variables
# Required: DATABASE_URL, AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, JWT_SECRET_KEY
```

3. **Initialize database:**
```bash
psql -U postgres -d your_database -f schema.sql
```

4. **Start the service:**
```bash
# Using uvicorn directly
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Or using Makefile
make run
```

## API Usage

### Query Endpoint

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "query": "What was ROAS last week for SB campaigns in Delhi?",
    "user_id": "user123",
    "session_id": "session456"
  }'
```

### Health Check

```bash
curl http://localhost:8000/api/v1/health
```

## Example Queries

1. **Simple ROAS query:**
   - "What was ROAS last week for SB campaigns?"

2. **Multi-metric comparison:**
   - "Compare ROAS, CPC, and CTR for SP vs SB vs SD last month"

3. **Product-level analysis:**
   - "Top 5 products by net sales in Delhi last month with ad spend"

4. **Time-series analysis:**
   - "Show spend trend for Meta campaigns over the last 30 days"

## Architecture Overview

```
User Query
    ↓
API Gateway (FastAPI)
    ↓
Orchestrator
    ↓
Router Agent → Task Spec
    ↓
Planner Agent → Execution Plan (DAG)
    ↓
Guardrail Agent → Validation
    ↓
Data Access Agents → Execute Steps
    ↓
Computation Agent → Calculate Metrics
    ↓
Answer Composer → Natural Language Answer
    ↓
Response
```

## Configuration

- **Tool Registry**: `config/tools.yaml` - Define data sources and capabilities
- **Metric Dictionary**: `config/metrics.yaml` - Define metric formulas
- **Schema Registry**: Auto-loaded from database or manual definitions

## Development

Run tests:
```bash
make test
```

Format code:
```bash
make format
```

Lint code:
```bash
make lint
```

## Production Deployment

1. Set `API_RELOAD=false` in `.env`
2. Use proper WSGI server (gunicorn with uvicorn workers)
3. Configure reverse proxy (nginx)
4. Set up monitoring and logging
5. Configure secrets management (Vault/AWS Secrets Manager)


# Hybrid MCP Architecture

## Overview

This system implements a **hybrid approach** combining template-based SQL generation with LLM-powered SQL generation using schema context (MCP-style). This provides the best of both worlds:

- **Flexibility**: LLM can generate complex SQL queries based on natural language
- **Safety**: Guardrails validate SQL before execution
- **Schema Awareness**: LLM has full context of table structures and relationships
- **Fallback**: Template-based queries still available for common patterns

## Architecture

```
User Query
    ↓
Router Agent → TaskSpec
    ↓
Planner Agent (with Schema Context)
    ├─→ Schema Context Service (provides table metadata)
    ├─→ Tool Registry (available capabilities)
    └─→ Metric Dictionary (metric definitions)
    ↓
Execution Plan (may include LLM-generated SQL)
    ↓
Guardrail Agent (validates SQL safety + schema)
    ↓
Data Access Agent
    ├─→ If SQL provided: Use LLM-generated SQL
    └─→ Else: Use template-based SQL
    ↓
Execute Query → Results
```

## Components

### 1. Schema Context Service (`app/services/schema_context.py`)

Provides schema metadata in formats optimized for LLM consumption:

**Key Methods:**
- `get_table_context(schema, table)`: Get detailed table information
- `get_relevant_tables_for_query(metrics, entities)`: Smart table selection based on query
- `format_for_llm(tables)`: Format schema for LLM prompts
- `build_mcp_tool_definition(schema, table)`: MCP-style tool definitions

**Example Output:**
```markdown
## Table: public.shopify_orders

**Columns:**
- `order_id` (text) [PRIMARY KEY] [NOT NULL]
- `created_at` (timestamp with time zone)
- `total_price_amount` (numeric)
- `ship_city` (text)

**Primary Keys:** order_id
**Date Column:** `created_at` (use for date filtering)
```

### 2. SQL Generator Agent (`app/agents/sql_generator.py`)

Standalone agent that generates SQL using LLM with schema context:

**Features:**
- Uses schema context to understand table structure
- Generates parameterized queries (prevents SQL injection)
- Returns structured JSON with SQL, params, and explanation

**Usage:**
```python
sql_generator = SQLGeneratorAgent()
result = sql_generator.generate_sql(
    query_description="Get revenue for Delhi last month",
    tables=["public.shopify_orders"],
    metrics=["revenue"],
    entities={"geo": "Delhi"},
    date_start="2025-10-01",
    date_end="2025-10-31"
)
# Returns: {"sql": "SELECT ...", "params": {...}, "explanation": "..."}
```

### 3. Enhanced Planner Agent (`app/agents/planner.py`)

Now includes schema context in prompts:

**Changes:**
- Builds schema catalog based on query context
- Includes relevant table schemas in system prompt
- Planner can generate SQL directly in execution plan
- Falls back to template-based if no SQL provided

**Example Plan Output:**
```json
{
  "steps": [
    {
      "id": "sales1",
      "tool": "sales_db.query",
      "inputs": {
        "sql": "SELECT order_id, total_price_amount as revenue FROM public.shopify_orders WHERE created_at >= :date_start AND created_at <= :date_end AND ship_city LIKE :geo",
        "params": {
          "date_start": "2025-10-01",
          "date_end": "2025-10-31",
          "geo": "%Delhi%"
        }
      }
    }
  ]
}
```

### 4. Enhanced Data Access Agents (`app/agents/data_access.py`)

Support both template-based and LLM-generated SQL:

**Logic:**
```python
if 'sql' in inputs:
    # Use LLM-generated SQL
    sql = inputs['sql']
    params = inputs.get('params', {})
else:
    # Use template-based SQL
    sql, params = self._build_query(query_inputs)
```

### 5. Enhanced Guardrail (`app/agents/guardrail.py`)

Validates LLM-generated SQL:

**Validations:**
- ✅ Blocks dangerous operations (DROP, DELETE, UPDATE, etc.)
- ✅ Warns about SELECT *
- ✅ Validates tables against schema registry
- ✅ Checks for parameterized queries
- ✅ Validates date ranges

## How It Works

### Step 1: Query Analysis

Router analyzes user query:
```
"What was revenue in Delhi last month?"
```

### Step 2: Schema Context Building

Planner identifies relevant tables:
- Query mentions "revenue" → `shopify_orders`
- Mentions "Delhi" → geo filtering needed
- Mentions "last month" → date filtering needed

Schema Context Service provides:
- Table structure for `public.shopify_orders`
- Column names, types, primary keys
- Date column information
- Foreign key relationships

### Step 3: SQL Generation

Planner (with schema context) generates:
```sql
SELECT 
    order_id,
    total_price_amount as revenue,
    created_at
FROM public.shopify_orders
WHERE created_at >= :date_start
  AND created_at <= :date_end
  AND ship_city LIKE :geo
```

### Step 4: Validation

Guardrail validates:
- ✅ No dangerous operations
- ✅ Tables exist in schema
- ✅ Uses parameterized queries
- ✅ Date range is reasonable

### Step 5: Execution

Data Access Agent executes:
```python
with engine.connect() as conn:
    result = pd.read_sql(text(sql), conn, params=params)
```

## Benefits

### 1. **Flexibility**
- LLM can generate complex queries with joins, aggregations
- Handles ad-hoc queries without template updates
- Adapts to schema changes automatically

### 2. **Safety**
- Guardrails block dangerous operations
- Parameterized queries prevent SQL injection
- Schema validation ensures correct table/column names

### 3. **Context-Aware**
- LLM understands table relationships
- Knows which columns to use for joins
- Understands date columns for filtering

### 4. **Hybrid Approach**
- Template-based for common patterns (faster)
- LLM-generated for complex queries (flexible)
- Best of both worlds

## Example Queries

### Simple Query (Template-based)
```
"What was revenue yesterday?"
```
→ Uses template-based query (fast, predictable)

### Complex Query (LLM-generated)
```
"Show me ROAS by campaign for Meta ads joined with sales data, filtered by Delhi, last 30 days"
```
→ LLM generates complex SQL with joins:
```sql
SELECT 
    m.campaign_id,
    m.campaign_name,
    m.spend,
    SUM(s.total_price_amount) as revenue,
    SUM(s.total_price_amount) / m.spend as roas
FROM public.dw_meta_ads_attribution m
LEFT JOIN public.shopify_orders s 
    ON m.campaign_id = s.campaign_id
WHERE m.date_start >= :date_start
  AND m.date_start <= :date_end
  AND s.ship_city LIKE :geo
GROUP BY m.campaign_id, m.campaign_name, m.spend
```

## Configuration

### Schema Context Service

Automatically loads schema from database on startup. Can also use fallback manual definitions.

### SQL Generator

Configured via Azure OpenAI settings:
- `azure_openai_endpoint`
- `azure_openai_deployment_name`
- `azure_openai_api_key`

Temperature set to 0.1 for deterministic SQL generation.

## Migration Path

### From Template-Only to Hybrid

1. **Phase 1**: Add schema context service ✅
2. **Phase 2**: Update planner to include schema in prompts ✅
3. **Phase 3**: Update data access agents to accept LLM SQL ✅
4. **Phase 4**: Enhance guardrails for SQL validation ✅
5. **Phase 5**: Monitor and tune (ongoing)

## Future Enhancements

1. **Query Optimization**: LLM can suggest index usage
2. **Query Caching**: Cache LLM-generated SQL for reuse
3. **Query Explanation**: Better explanations of generated SQL
4. **Error Recovery**: Auto-fix SQL errors with LLM feedback
5. **Multi-table Joins**: Better handling of complex joins

## Troubleshooting

### Issue: LLM generates invalid SQL

**Solution**: Guardrail will catch and reject. Check logs for validation errors.

### Issue: SQL uses wrong table/column names

**Solution**: Ensure schema registry is up to date. Check `schema_context.format_for_llm()` output.

### Issue: Performance issues with LLM-generated SQL

**Solution**: 
- Add query hints in prompts
- Use template-based for common patterns
- Cache frequent queries

## Best Practices

1. **Always use parameterized queries** (LLM is instructed to do this)
2. **Validate schema** before generating SQL
3. **Monitor query performance** and optimize as needed
4. **Keep schema registry updated** for accurate SQL generation
5. **Use templates for high-frequency queries** (faster)


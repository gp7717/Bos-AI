# Query Architecture & MCP Evaluation

## Current Query Architecture

### How Agents Make Queries

Your system uses a **template-based SQL generation** approach with specialized agents:

```
User Query
    ↓
Router Agent (LLM) → TaskSpec (normalized query)
    ↓
Planner Agent (LLM) → ExecutionPlan (DAG with SQL templates)
    ↓
Guardrail Agent → Validates plan & queries
    ↓
Data Access Agents → Generate & Execute SQL
    ↓
Computation Agent → Joins/Aggregations
    ↓
Answer Composer → Final response
```

### 1. **SQL Generation (Template-Based, Not Free-Form)**

**Current Approach:**
- **Schema Registry** provides table/column metadata
- **Data Access Agents** build SQL using **constrained templates**
- SQL is **parameterized** (prevents SQL injection)
- Column names validated against schema registry

**Example from `SalesDBAgent._build_query()`:**
```python
# Uses schema registry to get date column
date_column = schema_registry.get_date_column('public', 'shopify_orders')

# Builds parameterized query
base_query = """
    SELECT order_id, order_name, created_at, total_price_amount as revenue
    FROM public.shopify_orders o
    WHERE 1=1
    AND o.{date_column} >= :date_start
    AND o.{date_column} <= :date_end
"""
params = {'date_start': '2025-11-05', 'date_end': '2025-11-05'}
```

**Pros:**
- ✅ **Safe**: No SQL injection risk (parameterized queries)
- ✅ **Schema-aware**: Validates columns exist before querying
- ✅ **Type-safe**: Pydantic models ensure correct data types
- ✅ **Maintainable**: Changes to schema registry automatically reflected

**Cons:**
- ❌ **Limited flexibility**: Can't handle arbitrary SQL queries
- ❌ **Requires schema updates**: New tables/columns need registry updates
- ❌ **Template complexity**: Complex queries need more templates

### 2. **Query Execution Flow**

```python
# Orchestrator._execute_step()
1. Planner generates plan with tool="sales_db.query"
2. Orchestrator routes to SalesDBAgent
3. SalesDBAgent.execute(inputs) → builds SQL → executes via SQLAlchemy
4. Returns DataFrame with results
5. Computation agent processes results (joins/aggregations)
```

### 3. **Current Issues Identified**

**Problem: Row Loss (27 rows → 0 rows)**

Based on your logs, the issue is likely:

1. **Join Key Mismatch**: 
   - Sales DB returns 27 rows with columns like `order_id`, `revenue`, `created_at`
   - Computation tries to join on `campaign_id` or `date` 
   - If join keys don't exist or don't match → 0 rows

2. **Data Type Mismatch**:
   - Join keys might be different types (string vs int, date vs timestamp)
   - Pandas merge is strict about types

3. **Missing Columns**:
   - Computation expects columns that don't exist in result set
   - Formula evaluation fails silently

**Solution Applied:**
- ✅ Added detailed logging to track row counts at each step
- ✅ Added join key validation (checks if keys exist)
- ✅ Added warnings when joins result in 0 rows
- ✅ Added column logging for debugging

---

## MCP (Model Context Protocol) Evaluation

### What is MCP?

**Model Context Protocol** is a protocol by Anthropic that allows LLMs to:
- Connect to external data sources (databases, APIs, files)
- Discover available tools/resources at runtime
- Execute queries with structured interfaces

### MCP vs Current Architecture

| Aspect | Current Architecture | MCP Approach |
|--------|---------------------|--------------|
| **Query Generation** | Template-based SQL (constrained) | LLM generates SQL directly |
| **Schema Awareness** | Schema Registry (explicit) | Tool discovery (implicit) |
| **Safety** | Parameterized queries + validation | Depends on tool implementation |
| **Flexibility** | Limited to templates | Highly flexible |
| **Maintenance** | Schema registry updates needed | Tool registry updates |
| **Complexity** | Medium (explicit agents) | Low (standard protocol) |
| **Performance** | Optimized SQL templates | LLM-generated SQL (may be suboptimal) |

### Should You Use MCP?

**❌ Not Recommended for Your Use Case**

**Reasons:**

1. **You Already Have Specialized Agents**
   - Your agentic architecture is more sophisticated than MCP
   - MCP would add another abstraction layer without clear benefit

2. **Security Concerns**
   - MCP allows LLM to generate arbitrary SQL
   - Your current approach uses parameterized queries (safer)
   - Guardrails are easier to enforce with templates

3. **Schema Registry is Better**
   - Your schema registry provides explicit metadata
   - MCP tool discovery is implicit (less reliable)

4. **Performance**
   - Template-based SQL is optimized
   - LLM-generated SQL may be inefficient

5. **Complex Queries**
   - Your system handles multi-step DAGs with joins
   - MCP is better for simple tool calls

### When MCP Would Be Useful

MCP makes sense if:
- ✅ You need to query arbitrary databases/tools
- ✅ You want a standard protocol for tool integration
- ✅ You're building a general-purpose assistant
- ✅ You don't have domain-specific expertise

**Your system is domain-specific (analytics Q&A)**, so the current architecture is better.

---

## Recommended Improvements

### 1. **Enhanced Query Logging** ✅ (Applied)

- Log SQL queries at INFO level
- Log row counts at each step
- Log join operations with key validation
- Log column names for debugging

### 2. **Better Join Handling** (Next Steps)

**Current Issue:** Join keys might not match

**Solutions:**

```python
# Option A: Normalize join keys before join
def normalize_join_keys(left_data, right_data, join_keys):
    """Normalize data types and values for join keys."""
    for key in join_keys:
        # Convert to same type
        if left_data[key].dtype != right_data[key].dtype:
            # Convert both to string for fuzzy matching
            left_data[key] = left_data[key].astype(str)
            right_data[key] = right_data[key].astype(str)
        # Normalize whitespace/casing
        left_data[key] = left_data[key].str.strip().str.lower()
        right_data[key] = right_data[key].str.strip().str.lower()
```

### 3. **Fallback Strategies**

If join results in 0 rows:
1. Try fuzzy matching (normalize keys)
2. Try different join types (left → inner → full)
3. Warn user about data availability
4. Return partial results if possible

### 4. **Query Validation**

```python
def validate_join_feasibility(left_data, right_data, join_keys):
    """Check if join is likely to succeed."""
    for key in join_keys:
        left_values = set(left_data[key].unique())
        right_values = set(right_data[key].unique())
        overlap = left_values & right_values
        if len(overlap) == 0:
            logger.warning(f"No overlapping values for join key '{key}'")
```

---

## Debugging Your Current Issue

### Steps to Diagnose

1. **Check Logs** (now enhanced):
   ```
   🔍 [SALES_DB] Generated SQL | sql=...
   ✅ [SALES_DB] Query executed successfully | row_count=27
   🔗 [COMPUTATION] Starting join operation | left_rows=27 | right_rows=...
   ⚠️ [COMPUTATION] Join resulted in 0 rows!
   ```

2. **Verify Join Keys**:
   - Check if `campaign_id` exists in sales data
   - Check if `date` columns match format
   - Check data types (string vs int vs date)

3. **Check Planner Output**:
   - The Planner might be generating incorrect join keys
   - Review the execution plan DAG

### Quick Fix

If join keys don't match, you can:
1. **Modify Sales Query** to include join keys (if available)
2. **Use different join strategy** (e.g., date-based aggregation)
3. **Skip join** if not needed for the metric

---

## Conclusion

**Your current architecture is solid** for an analytics Q&A system:
- ✅ Safe (parameterized queries)
- ✅ Schema-aware (registry)
- ✅ Maintainable (explicit agents)

**MCP is not needed** - it would add complexity without clear benefits.

**Focus on fixing the join/computation logic** to prevent row loss, which is now easier with enhanced logging.


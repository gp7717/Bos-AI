# Quick Test Queries - Priority Order

## 🔴 Critical Tests (Must Pass)

### Test 1: API-Aware Selection (Your Original Issue)
```
Get the net profit graph for the last 4 months
```
**Expected**: Uses `api_docs` agent → `/api/net_profit` endpoint (NOT SQL)

### Test 2: Natural Language Decomposition
```
What's our performance this month?
```
**Expected**: Decomposed into specific sub-queries with inferred metrics and timeframes

### Test 3: Multiple Graphs
```
Show me sales trends and revenue breakdown as separate charts
```
**Expected**: Returns 2 graphs in response

### Test 4: Parallel Execution
```
Get sales data and API documentation for the orders endpoint
```
**Expected**: SQL/API and API docs agents execute in parallel

## 🟡 Important Tests (Should Pass)

### Test 5: Complex Multi-Part Query
```
Analyze our business performance: get sales data, calculate profit margins, compare to last month, and create visualizations
```
**Expected**: 4+ sub-queries, proper dependencies, multiple agents

### Test 6: Multiple API Metrics
```
What's our revenue, ROAS, and ad spend for this month?
```
**Expected**: Multiple API calls, parallel execution

### Test 7: Dependency Handling
```
Get net profit data, calculate growth rate, forecast next month, and show everything in graphs
```
**Expected**: Sequential execution for dependencies, parallel for independent parts

## 🟢 Advanced Tests (Nice to Have)

### Test 8: Very Complex Query
```
I need to see our sales performance for the last 6 months broken down by channel, compare it to the previous 6 months, calculate the growth rate, forecast the next 3 months, show trends, create breakdown charts, and also get the API documentation for the sales endpoint
```
**Expected**: Handles all parts correctly, multiple graphs, proper execution order

### Test 9: Mixed Sources
```
Get net profit from API and also query the orders table to see order details
```
**Expected**: API for net profit, SQL for orders, parallel execution

### Test 10: End-to-End Integration
```
I want a comprehensive business report: get net profit for last 3 months, calculate the average, compare to previous 3 months, show trends, create breakdown charts by channel, and also get me the API documentation
```
**Expected**: Full system test, all features working together

## Quick Test Commands

### Using Python
```python
from Agents.orchestrator.service import run
from Agents.core.models import AgentRequest

query = "Get the net profit graph for the last 4 months"
request = AgentRequest(question=query)
response = run(request)

# Check agent used
print(f"Agents used: {[r.agent for r in response.agent_results]}")
print(f"Graphs returned: {len(response.all_graphs)}")
```

### Using API (if server is running)
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Get the net profit graph for the last 4 months"}'
```

## Validation Checklist

For each test, check:
- ✅ Correct agent selected (API vs SQL)
- ✅ Query decomposed properly
- ✅ Tools identified correctly
- ✅ Dependencies handled
- ✅ Parallel execution (when independent)
- ✅ Multiple graphs (when requested)
- ✅ Response structure correct
- ✅ Performance improved


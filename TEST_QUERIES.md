# Complex Query Test Suite

This document contains complex test queries to validate the enhanced multi-agent system with:
- Natural language query decomposition
- Capability-aware agent selection
- Multiple graphs support
- Parallel processing
- Multiple data sources

## Test Categories

### Category 1: API-Aware Selection Tests
**Goal**: Verify system prefers API agent when metrics are available via API

1. **Simple Net Profit Query**
   ```
   Get the net profit graph for the last 4 months
   ```
   **Expected**: Should use `api_docs` agent (not SQL) since net profit is available via `/api/net_profit`

2. **Net Profit with Comparison**
   ```
   Show me net profit trends for Q3 and Q4, and compare them
   ```
   **Expected**: 
   - API agent for net profit data (2 calls or single call with date range)
   - Computation agent for comparison
   - Graph agent for visualization

3. **Multiple API Metrics**
   ```
   What's our revenue, ROAS, and ad spend for this month?
   ```
   **Expected**: 
   - API agent preferred for all metrics (revenue via `/api/sales`, ROAS via `/api/roas`, ad spend via `/api/ad_spend`)
   - Potentially parallel API calls

### Category 2: Natural Language Decomposition Tests
**Goal**: Verify generic queries are decomposed into specific sub-queries

4. **Vague Performance Query**
   ```
   What's our performance this month?
   ```
   **Expected**: 
   - Decomposed into: revenue, orders, net profit, trends
   - Multiple sub-queries with inferred timeframes
   - API agent preferred for available metrics

5. **Generic Trend Request**
   ```
   Show me the trends
   ```
   **Expected**:
   - Decomposed into: time-series data retrieval + visualization
   - Inferred metrics: revenue, orders (common business metrics)
   - Inferred timeframe: last 3 months (default)

6. **Comparison Without Details**
   ```
   Compare last quarter to this quarter
   ```
   **Expected**:
   - Decomposed into: Q3 data retrieval, Q4 data retrieval, comparison calculation, visualization
   - Dependencies: comparison depends on both data retrievals
   - Graph depends on comparison

### Category 3: Multiple Graphs Tests
**Goal**: Verify system generates multiple graphs when requested

7. **Explicit Multiple Graphs**
   ```
   Show me sales trends and revenue breakdown as separate charts
   ```
   **Expected**:
   - 2 graphs: line chart for trends, pie/donut chart for breakdown
   - Multiple sub-queries for different visualizations

8. **Implicit Multiple Visualizations**
   ```
   Display net profit over time and also show the channel breakdown
   ```
   **Expected**:
   - 2 graphs: line chart for trends, bar/pie chart for channel breakdown
   - API agent for net profit data (includes channel breakdown)

9. **Complex Multi-Graph Request**
   ```
   I need to see revenue trends, profit margins, and ROAS performance all visualized
   ```
   **Expected**:
   - 3 graphs: revenue trends, profit margins, ROAS
   - Multiple data sources (revenue, net profit, ROAS APIs)
   - Parallel data retrieval

### Category 4: Parallel Processing Tests
**Goal**: Verify independent agents execute in parallel

10. **Independent Data Sources**
    ```
    Get sales data and API documentation for the orders endpoint
    ```
    **Expected**:
    - SQL/API agent for sales data
    - API docs agent for documentation
    - Should execute in parallel (no dependencies)

11. **Mixed Independent Queries**
    ```
    Calculate the forecast for next quarter and also show me what API endpoints are available
    ```
    **Expected**:
    - Computation agent for forecast
    - API docs agent for endpoint listing
    - Parallel execution

12. **Multiple Independent Metrics**
    ```
    Fetch revenue, orders count, and ad spend data
    ```
    **Expected**:
    - Multiple API calls (if available) or SQL queries
    - Parallel execution when independent
    - All results combined

### Category 5: Complex Multi-Part Queries
**Goal**: Test complex queries requiring multiple agents and steps

13. **Full Analysis Request**
    ```
    Analyze our business performance: get sales data, calculate profit margins, compare to last month, and create visualizations
    ```
    **Expected**:
    - Decomposed into 4+ sub-queries
    - SQL/API → Computation → Comparison → Visualization
    - Dependencies properly handled

14. **Nested Dependencies**
    ```
    Get net profit data, calculate growth rate, forecast next month, and show everything in graphs
    ```
    **Expected**:
    - API agent for net profit
    - Computation for growth rate (depends on net profit)
    - Computation for forecast (depends on growth rate)
    - Graph agent (depends on all data)
    - Sequential execution for dependencies

15. **Multi-Source Comparison**
    ```
    Compare sales from our database with revenue from the API and show the difference
    ```
    **Expected**:
    - SQL agent for database sales
    - API agent for API revenue
    - Parallel execution (independent sources)
    - Computation for difference
    - Graph for visualization

### Category 6: Edge Cases and Error Handling

16. **Ambiguous Query**
    ```
    Tell me about our data
    ```
    **Expected**:
    - Should decompose into reasonable defaults
    - May ask for clarification or use heuristics

17. **Very Long Query**
    ```
    I need to see our sales performance for the last 6 months broken down by channel, compare it to the previous 6 months, calculate the growth rate, forecast the next 3 months, show trends, create breakdown charts, and also get the API documentation for the sales endpoint
    ```
    **Expected**:
    - Multiple sub-queries
    - Proper dependency resolution
    - Parallel execution where possible
    - Multiple graphs

18. **Mixed API and SQL Requirements**
    ```
    Get net profit from API and also query the orders table to see order details
    ```
    **Expected**:
    - API agent for net profit (preferred)
    - SQL agent for orders table (API doesn't have this)
    - Parallel execution

### Category 7: Time-Based Complex Queries

19. **Relative Time Inference**
    ```
    Show me this week's performance compared to last week
    ```
    **Expected**:
    - Inferred timeframes: current week, previous week
    - Two data retrieval sub-queries
    - Comparison computation
    - Visualization

20. **Multiple Time Periods**
    ```
    Display revenue for January, February, and March, and show the monthly trend
    ```
    **Expected**:
    - Three data retrieval sub-queries (or one with date range)
    - Trend analysis
    - Line chart visualization

### Category 8: Metric-Specific Tests

21. **ROAS Analysis**
    ```
    What's our ROAS performance and show it as a chart
    ```
    **Expected**:
    - API agent for ROAS (available via `/api/roas`)
    - Graph agent for visualization
    - Should NOT use SQL/computation

22. **COGS Breakdown**
    ```
    Get COGS data broken down by source and visualize it
    ```
   **Expected**:
   - API agent for COGS (available via `/api/cogs`)
   - Graph agent for breakdown visualization

23. **Multiple Financial Metrics**
    ```
    Show me revenue, profit, ROAS, and ad spend all together
    ```
    **Expected**:
    - Multiple API calls (all available via API)
    - Combined visualization or multiple graphs
    - Parallel API execution

### Category 9: Visualization-Specific Tests

24. **Chart Type Inference**
    ```
    Show me sales trends over time
    ```
    **Expected**:
    - Line chart (trends over time)
    - API agent for sales data

25. **Multiple Chart Types**
    ```
    Display revenue trends as a line chart and show the category breakdown as a pie chart
    ```
    **Expected**:
    - 2 graphs: line chart and pie chart
    - Different chart types for different data

### Category 10: Integration Tests

26. **End-to-End Complex Query**
    ```
    I want a comprehensive business report: get net profit for last 3 months, calculate the average, compare to previous 3 months, show trends, create breakdown charts by channel, and also get me the API documentation
    ```
    **Expected**:
    - Multiple sub-queries
    - API agent for net profit (preferred)
    - Computation for average and comparison
    - Multiple graphs (trends + breakdown)
    - API docs agent
    - Proper dependency handling
    - Parallel execution where possible

27. **Real-World Example**
    ```
    Get the net profit graph for the last 4 months
    ```
    **Expected** (from your example):
    - Should use API agent (not SQL)
    - Single API call to `/api/net_profit` with date range
    - Graph agent for visualization
    - Response includes graph data

## Test Execution Checklist

For each query, verify:

- [ ] Query is properly decomposed into sub-queries
- [ ] Correct agents are selected (API preferred when available)
- [ ] Tools are correctly identified from capability registry
- [ ] Dependencies are properly resolved
- [ ] Parallel execution occurs when agents are independent
- [ ] Multiple graphs are generated when requested
- [ ] Response includes all expected data
- [ ] Performance is improved (parallel execution)
- [ ] Error handling works gracefully

## Expected Performance Improvements

- **Sequential → Parallel**: 2-4x speedup for independent agents
- **SQL → API**: Faster when metrics available via API (pre-calculated)
- **Single → Multiple Graphs**: All graphs returned in response
- **Generic → Specific**: Better agent selection and tool usage

## Success Criteria

1. ✅ "Get net profit graph" uses API agent (not SQL)
2. ✅ Generic queries are decomposed into specific sub-queries
3. ✅ Multiple graphs are returned when requested
4. ✅ Independent agents execute in parallel
5. ✅ System handles complex multi-part queries correctly
6. ✅ Dependencies are respected (sequential when needed)
7. ✅ Capability registry correctly identifies available tools
8. ✅ Response time improved for complex queries


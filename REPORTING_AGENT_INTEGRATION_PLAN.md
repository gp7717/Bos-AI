# Reporting Agent Integration Plan

## Executive Summary

This document outlines the plan to integrate a **Reporting Agent** into the existing Bos-AI multi-agent system. The Reporting Agent will handle large data exports, create formatted tables, and generate CSV/Excel files on-demand without disrupting the existing workflow.

## Objectives

1. **On-Demand Activation**: Agent is called only when explicitly requested (via keywords or user intent)
2. **Non-Disruptive**: Does not interfere with existing agent execution flow
3. **Data Processing**: Handles large datasets from SQL, API Docs, or Computation agents
4. **Export Capabilities**: Generates CSV and Excel files when requested
5. **Table Formatting**: Creates well-formatted tables for presentation

## Architecture Overview

### Integration Pattern: Post-Processing Agent

The Reporting Agent will follow a **post-processing pattern**:
- Executes **after** data-fetching agents (SQL, API Docs, Computation)
- Operates on `TabularResult` data from previous agents
- Only activated when reporting keywords are detected or explicitly requested
- Returns file paths/URLs or formatted data in `AgentResult`

### System Flow Integration

```
┌─────────────────────────────────────────────────────────────┐
│                    Existing Flow (Unchanged)                 │
├─────────────────────────────────────────────────────────────┤
│  User Question → Planner → [SQL/API/Computation] → Composer  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │  Reporting Agent?   │
                    │  (Conditional)      │
                    └─────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │                   │
                    ▼                   ▼
            ┌──────────────┐   ┌──────────────┐
            │  Yes: Export │   │  No: Skip    │
            │  CSV/Excel   │   │  (Normal)    │
            └──────────────┘   └──────────────┘
```

## Design Decisions

### 1. Agent Registration

**Location**: `Agents/core/models.py`

**Changes Required**:
```python
# Add "reporting" to AgentName type
AgentName = Literal["sql", "computation", "planner", "composer", "api_docs", "reporting"]
```

### 2. Planner Integration

**Location**: `Agents/orchestrator/planner.py`

**Strategy**: Add reporting keywords to heuristic matching

**Keywords to Add**:
```python
_REPORTING_KEYWORDS = {
    "export",
    "csv",
    "excel",
    "download",
    "report",
    "file",
    "spreadsheet",
    "table",
    "format",
    "save",
    "generate report",
    "export data",
    "download data",
}
```

**Selection Logic**:
- Reporting agent should be selected **only** when:
  1. Reporting keywords are detected in question, AND
  2. At least one data-fetching agent (SQL/API/Computation) is also selected
- Reporting agent should execute **after** data-fetching agents

**Modified Heuristic**:
```python
def _heuristic_candidates(question: str) -> List[AgentName]:
    # ... existing logic ...
    reporting_score = sum(keyword in lowered for keyword in _REPORTING_KEYWORDS)
    
    # Reporting agent only if reporting keywords AND data agents present
    if reporting_score > 0 and (sql_score > 0 or api_score > 0 or comp_score > 0):
        # Add reporting agent AFTER data agents
        # (order will be: [sql/api/computation, reporting])
        pass
```

### 3. Orchestrator Workflow Integration

**Location**: `Agents/orchestrator/workflow.py`

**Changes Required**:
1. Add reporting agent case in `execute_agent()`:
```python
elif agent == "reporting":
    result = _run_reporting_agent(
        question=request.question,
        context=request.context,
        agent_results=state.get("agent_results", [])  # Access previous results
    )
```

**Key Design**: Reporting agent needs access to **previous agent results** to process their `TabularResult` data.

**State Access Pattern**:
- Reporting agent reads `agent_results` from state
- Extracts `TabularResult` from successful agents
- Processes and exports data
- Returns `AgentResult` with file paths or formatted data

### 4. Reporting Agent Implementation

**Location**: `Agents/ReportingAgent/`

**Directory Structure**:
```
Agents/ReportingAgent/
├── __init__.py
├── agent.py          # Main ReportingAgent class
├── formatter.py      # Table formatting utilities
├── exporter.py       # CSV/Excel export functionality
└── utils.py          # Helper functions
```

**Agent Interface**:
```python
class ReportingAgent:
    def invoke(
        self,
        question: str,
        *,
        context: Optional[dict] = None,
        agent_results: Optional[List[AgentResult]] = None,
    ) -> AgentResult:
        """
        Processes tabular data from previous agents and generates reports.
        
        Args:
            question: User question (may contain export format preferences)
            context: Request context (may contain output directory, file naming)
            agent_results: Results from previous agents (contains TabularResult)
        
        Returns:
            AgentResult with:
            - answer: Natural language summary of export
            - tabular: Optional (if table preview requested)
            - trace: Export events
        """
```

**Key Responsibilities**:
1. **Data Extraction**: Extract `TabularResult` from `agent_results`
2. **Format Detection**: Parse question for CSV/Excel preference
3. **File Generation**: Create CSV/Excel files
4. **Path Management**: Return file paths or URLs
5. **Table Formatting**: Optionally format tables for display

### 5. File Storage Strategy

**Options**:

**Option A: Local File System** (Recommended for MVP)
- Store files in configured directory (e.g., `exports/`)
- Return file paths in `AgentResult.answer`
- Context can specify `export_directory`

**Option B: Temporary Files**
- Use `tempfile` module
- Return temporary file paths
- Files cleaned up after response (or with TTL)

**Option C: Cloud Storage** (Future)
- Upload to S3/Azure Blob
- Return signed URLs
- Requires cloud credentials in context

**Initial Implementation**: Option A (Local File System)

**Context Keys**:
```python
context = {
    "export_directory": "/path/to/exports",  # Optional, default: "./exports"
    "export_filename": "report_2024-01-15",  # Optional, auto-generated if missing
    "export_format": "excel",  # Optional: "csv", "excel", or "both"
}
```

### 6. Dependencies

**New Dependencies** (add to `requirements.txt`):
```
pandas>=2.0.0          # Data manipulation and Excel export
openpyxl>=3.1.0        # Excel file generation (.xlsx)
xlsxwriter>=3.1.0      # Alternative Excel writer (optional)
```

## Implementation Steps

### Phase 1: Core Agent Structure

1. **Create Reporting Agent Directory**
   - `Agents/ReportingAgent/__init__.py`
   - `Agents/ReportingAgent/agent.py` (stub implementation)

2. **Update Models**
   - Add `"reporting"` to `AgentName` in `models.py`
   - Ensure `AgentResult` supports file paths in `answer` or new field

3. **Update Planner**
   - Add `_REPORTING_KEYWORDS` set
   - Modify `_heuristic_candidates()` to detect reporting intent
   - Ensure reporting agent comes after data agents

4. **Update Orchestrator**
   - Add reporting agent case in `execute_agent()`
   - Pass `agent_results` to reporting agent

### Phase 2: Export Functionality

1. **Implement CSV Export**
   - Use pandas `to_csv()`
   - Handle large datasets (chunking if needed)
   - Proper encoding (UTF-8 with BOM for Excel compatibility)

2. **Implement Excel Export**
   - Use pandas `to_excel()` with openpyxl
   - Format headers (bold, freeze panes)
   - Handle large datasets (multiple sheets if needed)

3. **File Naming**
   - Auto-generate filenames: `report_{timestamp}_{agent_name}.{ext}`
   - Support custom filenames from context
   - Sanitize filenames (remove invalid characters)

### Phase 3: Table Formatting

1. **Table Preview**
   - Optionally return formatted table in `TabularResult`
   - Limit rows for preview (e.g., first 100 rows)
   - Format numbers, dates appropriately

2. **Data Cleaning**
   - Handle None/NaN values
   - Format dates/timestamps
   - Truncate long text fields

### Phase 4: Integration & Testing

1. **End-to-End Testing**
   - Test with SQL agent results
   - Test with API Docs agent results
   - Test with Computation agent results
   - Test with multiple agents (multi-source export)

2. **Error Handling**
   - Handle missing data gracefully
   - Handle file write errors
   - Handle permission errors

3. **Performance Testing**
   - Test with large datasets (10K+ rows)
   - Test concurrent requests
   - Monitor memory usage

## Usage Examples

### Example 1: CSV Export Request

**User Question**: "Show me all orders from last month and export to CSV"

**Flow**:
1. Planner detects: SQL keywords ("orders", "last month") + Reporting keywords ("export", "csv")
2. Planner selects: `["sql", "reporting"]`
3. SQL agent executes → returns `TabularResult` with order data
4. Reporting agent executes → processes SQL result → generates CSV file
5. Composer aggregates → includes file path in final answer

**Response**:
```
Answer: "I've retrieved 1,234 orders from last month. The data has been exported to CSV file: exports/report_2024-01-15_14-30-22_sql.csv"
```

### Example 2: Excel Export Request

**User Question**: "Get sales data and create an Excel report"

**Flow**:
1. Planner detects: SQL keywords + Reporting keywords ("excel", "report")
2. Planner selects: `["sql", "reporting"]`
3. SQL agent → sales data
4. Reporting agent → Excel file with formatting
5. Composer → final answer with file path

### Example 3: Explicit Agent Selection

**User Question**: "Query products table and use reporting agent to export"

**Flow**:
- User can explicitly request: `prefer_agents=("sql", "reporting")`
- Planner respects preference
- Normal execution flow

## Error Handling

### Scenario 1: No Data Available

**Situation**: Reporting agent called but no `TabularResult` in previous agents

**Handling**:
```python
if not tabular_data:
    return AgentResult(
        agent="reporting",
        status=AgentExecutionStatus.failed,
        error=AgentError(
            message="No tabular data available for export",
            type="NoDataError"
        ),
        answer="Unable to generate report: no data was returned by previous agents."
    )
```

### Scenario 2: File Write Error

**Situation**: Permission denied or disk full

**Handling**:
```python
try:
    df.to_csv(file_path)
except (PermissionError, OSError) as e:
    return AgentResult(
        agent="reporting",
        status=AgentExecutionStatus.failed,
        error=AgentError(
            message=f"Failed to write file: {str(e)}",
            type="FileWriteError"
        )
    )
```

### Scenario 3: Large Dataset

**Situation**: Dataset too large for memory

**Handling**:
- Use chunking for CSV export
- For Excel: Split into multiple sheets (max 1M rows per sheet)
- Warn user if dataset is very large

## Performance Considerations

### Memory Management

- **Streaming Export**: For very large datasets (>100K rows), use chunked writing
- **Pandas Optimization**: Use `dtype` optimization to reduce memory
- **Garbage Collection**: Explicitly free DataFrames after export

### File Size Limits

- **CSV**: No practical limit (streaming)
- **Excel**: ~1M rows per sheet (Excel limit), use multiple sheets if needed
- **Warning Threshold**: Warn if dataset > 10K rows

### Concurrent Requests

- **File Naming**: Use UUID or timestamp to avoid collisions
- **Directory Locking**: Not required (each request gets unique filename)
- **Cleanup**: Optional cleanup of old files (TTL-based)

## Security Considerations

### File Access

- **Directory Validation**: Ensure export directory is writable and safe
- **Path Traversal**: Sanitize filenames to prevent directory traversal
- **File Permissions**: Set appropriate file permissions (read-only for others)

### Data Privacy

- **Sensitive Data**: Consider masking sensitive fields (if configured)
- **Access Control**: Export directory should have appropriate access controls
- **Audit Logging**: Log all export operations (via trace events)

## Configuration

### Environment Variables

```bash
# Optional: Default export directory
REPORTING_AGENT_EXPORT_DIR=./exports

# Optional: Default file format
REPORTING_AGENT_DEFAULT_FORMAT=excel

# Optional: Max rows per Excel sheet
REPORTING_AGENT_EXCEL_MAX_ROWS=1000000

# Optional: Enable file cleanup (TTL in hours)
REPORTING_AGENT_CLEANUP_TTL_HOURS=24
```

### Context Configuration

```python
context = {
    "export_directory": "/custom/path",
    "export_filename": "custom_name",
    "export_format": "csv",  # or "excel" or "both"
    "export_include_timestamp": True,
    "export_max_preview_rows": 100,
}
```

## Testing Strategy

### Unit Tests

1. **Formatter Tests**
   - Test table formatting
   - Test data cleaning
   - Test preview row limiting

2. **Exporter Tests**
   - Test CSV generation
   - Test Excel generation
   - Test file naming
   - Test error handling

3. **Agent Tests**
   - Test data extraction from `agent_results`
   - Test format detection from question
   - Test error scenarios

### Integration Tests

1. **End-to-End Flow**
   - SQL → Reporting
   - API Docs → Reporting
   - Computation → Reporting
   - Multiple agents → Reporting

2. **Planner Integration**
   - Test keyword detection
   - Test agent ordering
   - Test preference handling

3. **Orchestrator Integration**
   - Test state passing
   - Test result accumulation
   - Test composer aggregation

## Migration & Rollout

### Phase 1: Development (Week 1)
- Implement core agent structure
- Basic CSV export
- Integration with orchestrator

### Phase 2: Enhancement (Week 2)
- Excel export
- Table formatting
- Error handling

### Phase 3: Testing (Week 3)
- Unit tests
- Integration tests
- Performance testing

### Phase 4: Deployment (Week 4)
- Production deployment
- Monitoring
- Documentation

## Backward Compatibility

### No Breaking Changes

- Existing agents unchanged
- Existing workflows unchanged
- Reporting agent is **additive only**
- If reporting agent not selected, flow is identical to current behavior

### Optional Feature

- Reporting agent only activates when:
  1. Keywords detected, OR
  2. Explicitly requested via `prefer_agents`

## Future Enhancements

### Phase 2 Features

1. **Multiple Format Support**
   - PDF export
   - JSON export
   - Parquet export (for data science)

2. **Advanced Formatting**
   - Custom column formatting
   - Charts/graphs in Excel
   - Conditional formatting

3. **Cloud Storage Integration**
   - S3 upload
   - Azure Blob upload
   - Signed URL generation

4. **Scheduled Exports**
   - Cron-like scheduling
   - Email delivery
   - Webhook notifications

5. **Data Transformation**
   - Pivot tables
   - Aggregations
   - Filtering/sorting

## Success Metrics

### Functional Metrics

- ✅ Reporting agent activates only when requested
- ✅ CSV/Excel files generated correctly
- ✅ File paths returned in response
- ✅ No disruption to existing workflows

### Performance Metrics

- CSV export: < 1s for 10K rows
- Excel export: < 5s for 10K rows
- Memory usage: < 500MB for 100K rows

### Quality Metrics

- Zero breaking changes to existing system
- 100% backward compatibility
- Error rate < 1%

## Conclusion

This plan provides a comprehensive approach to integrating the Reporting Agent without disrupting the existing system. The agent will be:

1. **On-Demand**: Only activated when explicitly requested
2. **Non-Disruptive**: Does not change existing agent behavior
3. **Extensible**: Easy to add new export formats
4. **Robust**: Handles errors gracefully
5. **Performant**: Efficient for large datasets

The implementation follows existing patterns and integrates seamlessly with the orchestrator architecture.


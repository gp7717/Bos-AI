# Simple Export Solution (No Agent Required)

## Analysis

### Current System Capabilities

1. **Tabular Data Already Available**: 
   - `OrchestratorResponse.data` contains `TabularResult` with columns, rows, row_count
   - For regular queries like "Show me all orders from last month", the table data is **already in the response**
   - Frontend/client just needs to render `response.data` as a table

2. **Export Requirement**:
   - Only needed when user explicitly requests CSV/Excel export
   - Can be handled as post-processing in Composer

## Solution: Lightweight Export Utility

### Approach: Export Utility + Composer Integration

**No new agent needed!** Instead:
1. Create export utility functions (CSV/Excel generation)
2. Detect export keywords in Composer
3. Generate files when export requested + tabular data exists
4. Add file paths to response metadata

### Benefits

✅ **Simpler**: No agent registration, no planner changes, no orchestrator changes  
✅ **Faster**: Less code, faster implementation  
✅ **Maintainable**: Export logic isolated in utility module  
✅ **Non-disruptive**: Existing flow unchanged  

## Implementation Plan

### 1. Create Export Utility Module

**Location**: `Agents/orchestrator/export_utils.py`

```python
"""Utility functions for exporting tabular data to CSV/Excel."""

import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from Agents.core.models import TabularResult


def detect_export_intent(question: str) -> Optional[str]:
    """
    Detect if user wants CSV or Excel export from question.
    
    Returns:
        "csv", "excel", "both", or None
    """
    lowered = question.lower()
    
    has_csv = any(kw in lowered for kw in ["csv", "export to csv", "download csv"])
    has_excel = any(kw in lowered for kw in ["excel", "xlsx", "spreadsheet", "export to excel", "download excel"])
    has_export = any(kw in lowered for kw in ["export", "download", "save as"])
    
    if has_csv and has_excel:
        return "both"
    if has_csv:
        return "csv"
    if has_excel:
        return "excel"
    if has_export:
        return "csv"  # Default to CSV if just "export" mentioned
    return None


def export_tabular_data(
    tabular: TabularResult,
    export_format: str,
    export_directory: Optional[str] = None,
    filename_prefix: Optional[str] = None,
) -> dict[str, str]:
    """
    Export TabularResult to CSV/Excel files.
    
    Args:
        tabular: Tabular data to export
        export_format: "csv", "excel", or "both"
        export_directory: Directory to save files (default: "./exports")
        filename_prefix: Optional filename prefix
    
    Returns:
        Dict with file paths: {"csv": "...", "excel": "..."} (keys present based on format)
    """
    if not tabular or not tabular.rows:
        raise ValueError("No tabular data to export")
    
    # Determine export directory
    export_dir = Path(export_directory or "./exports")
    export_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = filename_prefix or "export"
    base_name = f"{prefix}_{timestamp}"
    
    # Convert to DataFrame
    df = pd.DataFrame(tabular.rows)
    
    file_paths = {}
    
    # Export CSV
    if export_format in ("csv", "both"):
        csv_path = export_dir / f"{base_name}.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")  # UTF-8 with BOM for Excel
        file_paths["csv"] = str(csv_path)
    
    # Export Excel
    if export_format in ("excel", "both"):
        excel_path = export_dir / f"{base_name}.xlsx"
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Data")
            # Format header row
            worksheet = writer.sheets["Data"]
            from openpyxl.styles import Font
            for cell in worksheet[1]:
                cell.font = Font(bold=True)
            worksheet.freeze_panes = "A2"
        file_paths["excel"] = str(excel_path)
    
    return file_paths


def format_export_message(file_paths: dict[str, str], export_format: str) -> str:
    """Format user-friendly message about exported files."""
    messages = []
    
    if "csv" in file_paths:
        messages.append(f"CSV file: {file_paths['csv']}")
    if "excel" in file_paths:
        messages.append(f"Excel file: {file_paths['excel']}")
    
    if not messages:
        return "Export completed."
    
    return "Files exported:\n" + "\n".join(f"- {msg}" for msg in messages)
```

### 2. Update Composer to Handle Exports

**Location**: `Agents/orchestrator/composer.py`

**Changes**:
- Import export utilities
- Detect export intent from question
- Generate files if export requested and tabular data exists
- Add file paths to metadata

```python
from .export_utils import detect_export_intent, export_tabular_data, format_export_message

class Composer:
    # ... existing code ...
    
    def compose(
        self,
        *,
        question: str,
        planner_rationale: str,
        agent_results: Iterable[AgentResult],
        metadata: Optional[dict] = None,
    ) -> OrchestratorResponse:
        results = list(agent_results)
        summaries = []
        tabular = self._select_tabular(results)
        
        # ... existing summarization code ...
        
        # Check for export intent
        export_format = detect_export_intent(question)
        export_files = {}
        
        if export_format and tabular:
            try:
                # Get export directory from context (if available)
                # Note: We'd need to pass context to compose, or get from request
                export_files = export_tabular_data(
                    tabular=tabular,
                    export_format=export_format,
                    # export_directory=context.get("export_directory"),
                    # filename_prefix=context.get("export_filename"),
                )
            except Exception as e:
                # Log error but don't fail the response
                logger.warning(f"Export failed: {e}")
        
        # Build metadata
        response_metadata = metadata or {}
        if export_files:
            response_metadata["export_files"] = export_files
            # Optionally update answer to mention files
            # (or let LLM handle it naturally)
        
        return OrchestratorResponse(
            answer=str(answer),
            data=tabular,  # Always include tabular data for table display
            agent_results=results,
            metadata=response_metadata,
        )
```

**Issue**: Composer doesn't have access to `request.context`. We need to pass it.

**Solution**: Update `compose_node` in `workflow.py` to pass context:

```python
def compose_node(state: OrchestratorState) -> OrchestratorState:
    request = state["request"]
    planner = state.get("planner")
    agent_results = state.get("agent_results", [])
    trace = list(state.get("trace", []))

    response = _COMPOSER.compose(
        question=request.question,
        planner_rationale=planner.rationale if planner else "",
        agent_results=agent_results,
        context=request.context,  # Pass context for export directory
        metadata={
            "confidence": planner.confidence if planner else None,
            "agents": [result.agent for result in agent_results],
        },
    )
    # ... rest of code ...
```

### 3. Update Composer Signature

```python
def compose(
    self,
    *,
    question: str,
    planner_rationale: str,
    agent_results: Iterable[AgentResult],
    context: Optional[dict] = None,  # Add context parameter
    metadata: Optional[dict] = None,
) -> OrchestratorResponse:
    # ... use context for export_directory ...
```

## Usage Examples

### Example 1: Regular Query (Table Display)

**User Question**: "Show me all orders from last month"

**Flow**:
1. SQL agent executes → returns `TabularResult`
2. Composer detects: No export keywords → no file generation
3. Response includes: `data: TabularResult` (table data)
4. Frontend renders table from `response.data`

**Response**:
```json
{
  "answer": "Found 1,234 orders from last month...",
  "data": {
    "columns": ["order_id", "date", "amount", ...],
    "rows": [...],
    "row_count": 1234
  }
}
```

### Example 2: Export Request

**User Question**: "Show me all orders from last month and export to CSV"

**Flow**:
1. SQL agent executes → returns `TabularResult`
2. Composer detects: "export to csv" → `export_format = "csv"`
3. Composer generates CSV file
4. Response includes: `data: TabularResult` (for table) + `metadata.export_files` (file path)

**Response**:
```json
{
  "answer": "Found 1,234 orders from last month. Files exported:\n- CSV file: exports/export_20240115_143022.csv",
  "data": {
    "columns": ["order_id", "date", "amount", ...],
    "rows": [...],
    "row_count": 1234
  },
  "metadata": {
    "export_files": {
      "csv": "exports/export_20240115_143022.csv"
    }
  }
}
```

## File Structure

```
Agents/orchestrator/
├── composer.py          # Updated to handle exports
├── export_utils.py      # NEW: Export utility functions
└── workflow.py          # Updated to pass context to composer
```

## Dependencies

Add to `requirements.txt`:
```
pandas>=2.0.0
openpyxl>=3.1.0
```

## Comparison: Agent vs Utility

| Aspect | Agent Approach | Utility Approach |
|--------|---------------|------------------|
| **Code Complexity** | High (agent registration, planner, orchestrator) | Low (utility + composer update) |
| **Implementation Time** | 2-3 days | 1 day |
| **Maintenance** | More complex | Simpler |
| **Extensibility** | Easy to add features | Easy to add features |
| **Consistency** | Follows agent pattern | Simpler pattern |
| **Performance** | Slightly more overhead | Minimal overhead |

## Recommendation

**Use Utility Approach** because:
1. ✅ Export is a **post-processing step**, not a data-fetching agent
2. ✅ Simpler implementation and maintenance
3. ✅ No need for agent selection logic
4. ✅ Tabular data already available in Composer
5. ✅ Can be easily extended later if needed

## Future Extensions

If we need more complex reporting features later (scheduled exports, email delivery, cloud storage), we can:
1. Keep utility approach for simple exports
2. Add Reporting Agent for complex features
3. Or migrate utility to agent if complexity grows

## Implementation Steps

1. ✅ Create `export_utils.py` with export functions
2. ✅ Update `composer.py` to detect export intent and generate files
3. ✅ Update `workflow.py` to pass context to composer
4. ✅ Add dependencies (`pandas`, `openpyxl`)
5. ✅ Test with regular queries (table display)
6. ✅ Test with export requests (CSV/Excel)
7. ✅ Handle errors gracefully

## Testing

### Test Case 1: Regular Query
```python
request = AgentRequest(question="Show me all orders from last month")
response = run(request)
assert response.data is not None  # Table data present
assert "export_files" not in response.metadata  # No export
```

### Test Case 2: CSV Export
```python
request = AgentRequest(question="Show me all orders and export to CSV")
response = run(request)
assert response.data is not None  # Table data present
assert "csv" in response.metadata.get("export_files", {})  # CSV file generated
```

### Test Case 3: Excel Export
```python
request = AgentRequest(question="Get sales data and create Excel report")
response = run(request)
assert "excel" in response.metadata.get("export_files", {})  # Excel file generated
```

## Conclusion

**No agent needed!** A simple utility function integrated into Composer is sufficient and cleaner for this use case.


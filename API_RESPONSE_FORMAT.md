# API Response Format Specification for Ask Seleric

## Overview
This document defines the standard data format for API responses from the Ask Seleric endpoint (`/api/ask-seleric/query`).

## Complete Response Structure

```json
{
  "answer": "Text response explaining the query results",
  "data": {
    "columns": ["column1", "column2", ...],
    "rows": [
      { "column1": "value1", "column2": "value2", ... },
      ...
    ],
    "row_count": 7
  },
  "graph": {
    "chart_type": "bar",
    "x_axis": "weekday_name",
    "y_axis": ["avg_orders_per_day"],
    "data": [
      {
        "weekday_name": "Sunday",
        "avg_orders_per_day": 15.4,
        "total_orders_for_weekday": 77,
        "occurrences_in_month": 5
      },
      ...
    ],
    "title": "Average Orders per Day by Weekday (This Month)",
    "x_label": "Weekday (Sunday → Saturday)",
    "y_label": "Average Orders per Day",
    "trend_analysis": "Optional analysis text describing trends in the data"
  },
  "metadata": {
    "confidence": 0.9,
    "agents": ["sql", "graph"]
  },
  "trace": [...]
}
```

## Graph Object Specification

### Required Fields

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `chart_type` | `string` | Type of chart to render. Valid values: `"bar"`, `"line"`, `"area"`, `"pie"`, `"donut"` | `"bar"` |
| `x_axis` | `string` | Key name in data objects that represents the x-axis/category | `"weekday_name"` |
| `y_axis` | `string` or `string[]` | **RECOMMENDED: Use array format** - Key name(s) in data objects that represent y-axis values | `["avg_orders_per_day"]` or `"avg_orders_per_day"` |
| `data` | `object[]` | Array of data objects. Each object must contain keys matching `x_axis` and all `y_axis` values | See example below |

### Optional Fields

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `title` | `string` | Chart title displayed above the chart | `"Average Orders per Day by Weekday"` |
| `x_label` | `string` | Label for x-axis | `"Weekday"` |
| `y_label` | `string` | Label for y-axis (only used if single y-axis metric) | `"Average Orders per Day"` |
| `trend_analysis` | `string` | Optional text analysis displayed below the chart | `"Thursday shows a clear spike..."` |

## Recommended Format (Best Practice)

### ✅ **RECOMMENDED: Use Array Format for `y_axis`**

```json
{
  "graph": {
    "chart_type": "bar",
    "x_axis": "weekday_name",
    "y_axis": ["avg_orders_per_day"],  // ← Array format (recommended)
    "data": [
      {
        "weekday_name": "Sunday",
        "avg_orders_per_day": 15.4
      },
      {
        "weekday_name": "Monday",
        "avg_orders_per_day": 17.5
      }
    ],
    "title": "Average Orders per Day by Weekday",
    "x_label": "Weekday",
    "y_label": "Average Orders per Day"
  }
}
```

### ✅ **ACCEPTABLE: String Format for Single Metric**

```json
{
  "graph": {
    "chart_type": "bar",
    "x_axis": "weekday_name",
    "y_axis": "avg_orders_per_day",  // ← String format (works but not recommended)
    "data": [...]
  }
}
```

### ✅ **MULTIPLE Y-AXIS METRICS (Must Use Array)**

```json
{
  "graph": {
    "chart_type": "line",
    "x_axis": "date",
    "y_axis": ["revenue", "orders", "profit"],  // ← Multiple metrics
    "data": [
      {
        "date": "2025-11-01",
        "revenue": 10000,
        "orders": 50,
        "profit": 2000
      },
      ...
    ],
    "title": "Revenue, Orders, and Profit Over Time"
  }
}
```

## Data Object Requirements

Each object in the `data` array must:
1. **Contain the `x_axis` key** with a value (string, number, or date)
2. **Contain all `y_axis` keys** with numeric values (strings will be parsed to numbers)
3. **Can contain additional keys** (these will be ignored for charting but available in tooltips)

### Example Data Object

```json
{
  "weekday_name": "Sunday",           // ← Must match x_axis
  "avg_orders_per_day": 15.4,         // ← Must match y_axis[0]
  "total_orders_for_weekday": 77,     // ← Additional data (optional)
  "occurrences_in_month": 5           // ← Additional data (optional)
}
```

## Chart Type Specifics

### Bar Charts (`chart_type: "bar"`)
- Best for: Comparing values across categories
- Supports: Single or multiple y-axis metrics
- Example: Weekly sales comparison, category breakdowns

### Line Charts (`chart_type: "line"`)
- Best for: Trends over time
- Supports: Single or multiple y-axis metrics
- Example: Revenue trends, order volume over time

### Area Charts (`chart_type: "area"`)
- Best for: Cumulative trends or stacked data
- Supports: Single or multiple y-axis metrics
- Example: Cumulative revenue, stacked metrics

### Pie/Donut Charts (`chart_type: "pie"` or `"donut"`)
- Best for: Proportional breakdowns
- **Note**: Only uses the **first** y-axis metric
- Example: Category distribution, percentage breakdowns

## Validation Rules

1. ✅ `data` must be a non-empty array
2. ✅ Each data object must contain the `x_axis` key
3. ✅ Each data object must contain all keys specified in `y_axis` array
4. ✅ `y_axis` values should be numeric (strings will be parsed)
5. ✅ `chart_type` must be one of: `"bar"`, `"line"`, `"area"`, `"pie"`, `"donut"`

## Error Handling

The frontend will handle:
- Missing `y_axis` → defaults to empty array `[]`
- String `y_axis` → automatically converted to array `[y_axis]`
- Missing `x_axis` values → defaults to empty string `""`
- Non-numeric `y_axis` values → converted to `0`
- Invalid `chart_type` → defaults to `"line"`

## Complete Example Response

```json
{
  "answer": "Graph generated: Average Orders per Day by Weekday (This Month)\n\nSunday — 15.40 orders/day\nMonday — 17.50 orders/day\n...",
  "data": {
    "columns": ["dow", "weekday_name", "avg_orders_per_day", "total_orders_for_weekday", "occurrences_in_month"],
    "rows": [
      {
        "dow": 0,
        "weekday_name": "Sunday",
        "avg_orders_per_day": "15.40",
        "total_orders_for_weekday": "77",
        "occurrences_in_month": 5
      }
    ],
    "row_count": 7
  },
  "graph": {
    "chart_type": "bar",
    "x_axis": "weekday_name",
    "y_axis": ["avg_orders_per_day"],
    "data": [
      {
        "dow": 0,
        "weekday_name": "Sunday",
        "avg_orders_per_day": 15.4,
        "total_orders_for_weekday": 77,
        "occurrences_in_month": 5
      },
      {
        "dow": 1,
        "weekday_name": "Monday",
        "avg_orders_per_day": 17.5,
        "total_orders_for_weekday": 70,
        "occurrences_in_month": 4
      }
    ],
    "title": "Average Orders per Day by Weekday (This Month)",
    "x_label": "Weekday (Sunday → Saturday)",
    "y_label": "Average Orders per Day",
    "trend_analysis": "Thursday shows a clear spike (26.75 avg), the highest of the week..."
  },
  "metadata": {
    "confidence": 0.9,
    "agents": ["sql", "graph"]
  }
}
```

## Summary

**✅ RECOMMENDED STANDARD FORMAT:**
- Use **array format** for `y_axis`: `["metric_name"]` even for single metrics
- This ensures consistency and future-proofing for multiple metrics
- Makes the code more predictable and easier to maintain

**⚠️ ACCEPTABLE BUT NOT RECOMMENDED:**
- String format `"metric_name"` works but requires normalization in frontend
- Less flexible for future enhancements


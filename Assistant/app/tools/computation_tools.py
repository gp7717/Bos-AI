"""LangChain tools for computation operations."""
from typing import Dict, Any, List, Optional
from langchain_core.tools import tool
from pydantic import BaseModel, Field
import pandas as pd
from app.agents.computation import ComputationAgent
from app.config.logging_config import get_logger

logger = get_logger(__name__)

# Global computation agent instance
_computation_agent = ComputationAgent()


class ComputeMetricInput(BaseModel):
    """Input schema for metric computation tool."""
    metric_id: str = Field(description="Metric identifier (e.g., 'roas', 'cpc')")
    data: List[Dict[str, Any]] = Field(description="Input data as list of dictionaries")


class JoinDataInput(BaseModel):
    """Input schema for data join tool."""
    left_data: List[Dict[str, Any]] = Field(description="Left dataframe as list of dictionaries")
    right_data: List[Dict[str, Any]] = Field(description="Right dataframe as list of dictionaries")
    join_keys: List[str] = Field(description="Column names to join on")
    how: str = Field("left", description="Join type: 'left', 'right', 'inner', 'outer'")


class AggregateDataInput(BaseModel):
    """Input schema for data aggregation tool."""
    data: List[Dict[str, Any]] = Field(description="Input data as list of dictionaries")
    group_by: List[str] = Field(description="Columns to group by")
    aggregations: Dict[str, str] = Field(description="Aggregation functions: {column: 'sum'|'mean'|'count'|'max'|'min'}")


@tool(args_schema=ComputeMetricInput)
def create_compute_metric_tool(
    metric_id: str,
    data: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Compute a metric from data using metric definitions.
    
    Supports various metric types:
    - Ratio metrics (e.g., ROAS = revenue / spend)
    - Aggregation metrics (sum, mean, count, etc.)
    - Delta metrics (period-over-period changes)
    
    Args:
        metric_id: Metric identifier (e.g., 'roas', 'cpc', 'ctr')
        data: Input data as list of dictionaries (will be converted to DataFrame)
        
    Returns:
        Dictionary with 'data' (list of records), 'row_count', and 'columns'
    """
    logger.info(
        f"🧮 [COMPUTE_METRIC_TOOL] Computing metric | "
        f"metric_id={metric_id} | input_rows={len(data)}"
    )
    
    try:
        df = pd.DataFrame(data)
        
        if df.empty:
            logger.warning(f"⚠️ [COMPUTE_METRIC_TOOL] Empty input data")
            return {
                "data": [],
                "row_count": 0,
                "columns": [],
                "error": "Input data is empty"
            }
        
        result_df = _computation_agent.compute(metric_id, df)
        
        result = {
            "data": result_df.to_dict('records'),
            "row_count": len(result_df),
            "columns": list(result_df.columns)
        }
        
        logger.info(
            f"✅ [COMPUTE_METRIC_TOOL] Metric computed | "
            f"metric_id={metric_id} | result_rows={result['row_count']}"
        )
        
        return result
    except Exception as e:
        logger.error(
            f"❌ [COMPUTE_METRIC_TOOL] Computation failed | "
            f"metric_id={metric_id} | error={str(e)}",
            exc_info=True
        )
        return {
            "data": [],
            "row_count": 0,
            "columns": [],
            "error": str(e)
        }


@tool(args_schema=JoinDataInput)
def create_join_data_tool(
    left_data: List[Dict[str, Any]],
    right_data: List[Dict[str, Any]],
    join_keys: List[str],
    how: str = "left",
) -> Dict[str, Any]:
    """
    Join two datasets on specified keys.
    
    Args:
        left_data: Left dataframe as list of dictionaries
        right_data: Right dataframe as list of dictionaries
        join_keys: Column names to join on (can be same column name in both)
        how: Join type ('left', 'right', 'inner', 'outer')
        
    Returns:
        Dictionary with 'data' (list of records), 'row_count', and 'columns'
    """
    logger.info(
        f"🔗 [JOIN_DATA_TOOL] Joining data | "
        f"left_rows={len(left_data)} | right_rows={len(right_data)} | "
        f"join_keys={join_keys} | how={how}"
    )
    
    try:
        left_df = pd.DataFrame(left_data)
        right_df = pd.DataFrame(right_data)
        
        if left_df.empty:
            logger.warning(f"⚠️ [JOIN_DATA_TOOL] Left dataframe is empty")
            return {
                "data": [],
                "row_count": 0,
                "columns": [],
                "error": "Left dataframe is empty"
            }
        
        result_df = _computation_agent.join(left_df, right_df, join_keys, how=how)
        
        result = {
            "data": result_df.to_dict('records'),
            "row_count": len(result_df),
            "columns": list(result_df.columns)
        }
        
        logger.info(
            f"✅ [JOIN_DATA_TOOL] Join completed | "
            f"result_rows={result['row_count']}"
        )
        
        return result
    except Exception as e:
        logger.error(
            f"❌ [JOIN_DATA_TOOL] Join failed | error={str(e)}",
            exc_info=True
        )
        return {
            "data": [],
            "row_count": 0,
            "columns": [],
            "error": str(e)
        }


@tool(args_schema=AggregateDataInput)
def create_aggregate_data_tool(
    data: List[Dict[str, Any]],
    group_by: List[str],
    aggregations: Dict[str, str],
) -> Dict[str, Any]:
    """
    Aggregate data by grouping columns and applying aggregation functions.
    
    Args:
        data: Input data as list of dictionaries
        group_by: Columns to group by
        aggregations: Dictionary mapping columns to aggregation functions
                     (e.g., {'revenue': 'sum', 'orders': 'count'})
        
    Returns:
        Dictionary with 'data' (list of records), 'row_count', and 'columns'
    """
    logger.info(
        f"📊 [AGGREGATE_DATA_TOOL] Aggregating data | "
        f"input_rows={len(data)} | group_by={group_by} | "
        f"aggregations={aggregations}"
    )
    
    try:
        df = pd.DataFrame(data)
        
        if df.empty:
            logger.warning(f"⚠️ [AGGREGATE_DATA_TOOL] Input data is empty")
            return {
                "data": [],
                "row_count": 0,
                "columns": [],
                "error": "Input data is empty"
            }
        
        result_df = _computation_agent.aggregate(df, group_by, aggregations)
        
        result = {
            "data": result_df.to_dict('records'),
            "row_count": len(result_df),
            "columns": list(result_df.columns)
        }
        
        logger.info(
            f"✅ [AGGREGATE_DATA_TOOL] Aggregation completed | "
            f"result_rows={result['row_count']}"
        )
        
        return result
    except Exception as e:
        logger.error(
            f"❌ [AGGREGATE_DATA_TOOL] Aggregation failed | error={str(e)}",
            exc_info=True
        )
        return {
            "data": [],
            "row_count": 0,
            "columns": [],
            "error": str(e)
        }


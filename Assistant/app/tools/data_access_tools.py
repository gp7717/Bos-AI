"""LangChain tools for data access operations."""
from typing import Dict, Any, Optional
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from app.agents.data_access import get_data_access_agent
from app.config.logging_config import get_logger

logger = get_logger(__name__)


class SalesDBQueryInput(BaseModel):
    """Input schema for Sales DB query tool."""
    date_start: Optional[str] = Field(None, description="Start date (YYYY-MM-DD)")
    date_end: Optional[str] = Field(None, description="End date (YYYY-MM-DD)")
    geo: Optional[str] = Field(None, description="Geographic filter (city/region)")
    limit: int = Field(10000, description="Maximum number of rows to return")
    sql: Optional[str] = Field(None, description="Optional SQL query (LLM-generated)")
    params: Optional[Dict[str, Any]] = Field(None, description="SQL parameters")


class AmazonAdsDBQueryInput(BaseModel):
    """Input schema for Amazon Ads DB query tool."""
    date_start: Optional[str] = Field(None, description="Start date (YYYY-MM-DD)")
    date_end: Optional[str] = Field(None, description="End date (YYYY-MM-DD)")
    channel: Optional[str] = Field(None, description="Channel filter (SP, SB, SD)")
    limit: int = Field(100000, description="Maximum number of rows to return")
    sql: Optional[str] = Field(None, description="Optional SQL query (LLM-generated)")
    params: Optional[Dict[str, Any]] = Field(None, description="SQL parameters")


class MetaAdsDBQueryInput(BaseModel):
    """Input schema for Meta Ads DB query tool."""
    date_start: Optional[str] = Field(None, description="Start date (YYYY-MM-DD)")
    date_end: Optional[str] = Field(None, description="End date (YYYY-MM-DD)")
    limit: int = Field(100000, description="Maximum number of rows to return")
    sql: Optional[str] = Field(None, description="Optional SQL query (LLM-generated)")
    params: Optional[Dict[str, Any]] = Field(None, description="SQL parameters")


@tool(args_schema=SalesDBQueryInput)
def create_sales_db_tool(
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
    geo: Optional[str] = None,
    limit: int = 10000,
    sql: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Query the sales database for order and revenue data.
    
    Use this tool to fetch sales data from Shopify orders, including:
    - Order details (order_id, order_name, created_at)
    - Revenue metrics (total_price_amount)
    - Geographic data (ship_city, ship_country)
    - Product information (via joins with line items and variants)
    
    Args:
        date_start: Start date for filtering orders (YYYY-MM-DD)
        date_end: End date for filtering orders (YYYY-MM-DD)
        geo: Geographic filter (city name)
        limit: Maximum rows to return
        sql: Optional SQL query (if LLM generates SQL directly)
        params: SQL parameters for parameterized queries
        
    Returns:
        Dictionary with 'data' (list of records), 'row_count', and 'columns'
    """
    logger.info(
        f"📊 [SALES_DB_TOOL] Executing query | "
        f"date_start={date_start} | date_end={date_end} | geo={geo}"
    )
    
    agent = get_data_access_agent("sales_db")
    if not agent:
        return {
            "data": [],
            "row_count": 0,
            "columns": [],
            "error": "Sales DB agent not available"
        }
    
    inputs = {
        "inputs": {
            "date_start": date_start,
            "date_end": date_end,
            "geo": geo,
            "limit": limit,
        }
    }
    
    if sql:
        inputs["sql"] = sql
    if params:
        inputs["params"] = params
    
    try:
        result = agent.execute(inputs)
        logger.info(
            f"✅ [SALES_DB_TOOL] Query completed | "
            f"row_count={result.get('row_count', 0)}"
        )
        return result
    except Exception as e:
        logger.error(
            f"❌ [SALES_DB_TOOL] Query failed | error={str(e)}",
            exc_info=True
        )
        return {
            "data": [],
            "row_count": 0,
            "columns": [],
            "error": str(e)
        }


@tool(args_schema=AmazonAdsDBQueryInput)
def create_amazon_ads_db_tool(
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
    channel: Optional[str] = None,
    limit: int = 100000,
    sql: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Query the Amazon Ads database for campaign performance data.
    
    Use this tool to fetch Amazon advertising metrics including:
    - Campaign performance (spend, clicks, impressions)
    - Sales and orders attributed to ads
    - ROAS, ACOS, CTR, CPC metrics
    
    Args:
        date_start: Start date for filtering (YYYY-MM-DD)
        date_end: End date for filtering (YYYY-MM-DD)
        channel: Channel filter (SP, SB, SD)
        limit: Maximum rows to return
        sql: Optional SQL query (if LLM generates SQL directly)
        params: SQL parameters for parameterized queries
        
    Returns:
        Dictionary with 'data' (list of records), 'row_count', and 'columns'
    """
    logger.info(
        f"📊 [AMAZON_ADS_DB_TOOL] Executing query | "
        f"date_start={date_start} | date_end={date_end} | channel={channel}"
    )
    
    agent = get_data_access_agent("amazon_ads_db")
    if not agent:
        return {
            "data": [],
            "row_count": 0,
            "columns": [],
            "error": "Amazon Ads DB agent not available"
        }
    
    inputs = {
        "inputs": {
            "date_start": date_start,
            "date_end": date_end,
            "channel": channel,
            "limit": limit,
        }
    }
    
    if sql:
        inputs["sql"] = sql
    if params:
        inputs["params"] = params
    
    try:
        result = agent.execute(inputs)
        logger.info(
            f"✅ [AMAZON_ADS_DB_TOOL] Query completed | "
            f"row_count={result.get('row_count', 0)}"
        )
        return result
    except Exception as e:
        logger.error(
            f"❌ [AMAZON_ADS_DB_TOOL] Query failed | error={str(e)}",
            exc_info=True
        )
        return {
            "data": [],
            "row_count": 0,
            "columns": [],
            "error": str(e)
        }


@tool(args_schema=MetaAdsDBQueryInput)
def create_meta_ads_db_tool(
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
    limit: int = 100000,
    sql: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Query the Meta Ads database for campaign performance data.
    
    Use this tool to fetch Meta (Facebook) advertising metrics including:
    - Campaign performance (spend, clicks, impressions)
    - Attributed orders and revenue
    - CTR, CPC, CPM metrics
    
    Args:
        date_start: Start date for filtering (YYYY-MM-DD)
        date_end: End date for filtering (YYYY-MM-DD)
        limit: Maximum rows to return
        sql: Optional SQL query (if LLM generates SQL directly)
        params: SQL parameters for parameterized queries
        
    Returns:
        Dictionary with 'data' (list of records), 'row_count', and 'columns'
    """
    logger.info(
        f"📊 [META_ADS_DB_TOOL] Executing query | "
        f"date_start={date_start} | date_end={date_end}"
    )
    
    agent = get_data_access_agent("meta_ads_db")
    if not agent:
        return {
            "data": [],
            "row_count": 0,
            "columns": [],
            "error": "Meta Ads DB agent not available"
        }
    
    inputs = {
        "inputs": {
            "date_start": date_start,
            "date_end": date_end,
            "limit": limit,
        }
    }
    
    if sql:
        inputs["sql"] = sql
    if params:
        inputs["params"] = params
    
    try:
        result = agent.execute(inputs)
        logger.info(
            f"✅ [META_ADS_DB_TOOL] Query completed | "
            f"row_count={result.get('row_count', 0)}"
        )
        return result
    except Exception as e:
        logger.error(
            f"❌ [META_ADS_DB_TOOL] Query failed | error={str(e)}",
            exc_info=True
        )
        return {
            "data": [],
            "row_count": 0,
            "columns": [],
            "error": str(e)
        }


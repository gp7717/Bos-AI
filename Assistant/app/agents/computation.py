"""Computation Agent - Metric calculations and aggregations."""
from typing import Dict, Any, List, Optional
import pandas as pd
from app.services.metric_dictionary import metric_dictionary
from app.models.schemas import MetricDefinition
from app.config.logging_config import get_logger

logger = get_logger(__name__)


class ComputationAgent:
    """Computes metrics and performs aggregations."""
    
    def __init__(self):
        """Initialize computation agent."""
        pass
    
    def compute(self, metric_id: str, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Compute a metric from data."""
        import time
        start_time = time.time()
        
        logger.info(
            f"📥 [COMPUTATION] Input | "
            f"metric_id={metric_id} | "
            f"input_rows={len(data)} | "
            f"input_columns={list(data.columns) if len(data) > 0 else []} | "
            f"kwargs={kwargs}"
        )
        
        logger.info(
            f"🧮 [COMPUTATION] Computing metric | "
            f"metric_id={metric_id} | input_rows={len(data)} | "
            f"input_columns={list(data.columns) if len(data) > 0 else []}"
        )
        
        metric_def = metric_dictionary.get_metric(metric_id)
        if not metric_def:
            logger.error(f"❌ [COMPUTATION] Unknown metric | metric_id={metric_id}")
            raise ValueError(f"Unknown metric: {metric_id}")
        
        # Compute based on type
        try:
            if metric_def.computation_type == 'ratio':
                result = self._compute_ratio(metric_def, data, **kwargs)
            elif metric_def.computation_type == 'aggregation':
                result = self._compute_aggregation(metric_def, data, **kwargs)
            elif metric_def.computation_type == 'delta':
                result = self._compute_delta(metric_def, data, **kwargs)
            else:
                result = self._compute_simple(metric_def, data, **kwargs)
            
            logger.info(
                f"✅ [COMPUTATION] Metric computed | "
                f"metric_id={metric_id} | "
                f"result_rows={len(result)} | "
                f"result_columns={list(result.columns)}"
            )
            
            if len(result) == 0 and len(data) > 0:
                logger.warning(
                    f"⚠️ [COMPUTATION] Computation resulted in 0 rows! | "
                    f"metric_id={metric_id} | input_rows={len(data)} | "
                    f"computation_type={metric_def.computation_type}"
                )
            
            # Log outputs
            execution_time_ms = (time.time() - start_time) * 1000
            logger.info(
                f"📤 [COMPUTATION] Output | "
                f"metric_id={metric_id} | "
                f"result_rows={len(result)} | "
                f"result_columns={list(result.columns)} | "
                f"execution_time_ms={execution_time_ms:.2f}"
            )
            
            return result
        except Exception as e:
            logger.error(
                f"❌ [COMPUTATION] Metric computation failed | "
                f"metric_id={metric_id} | error={str(e)}",
                exc_info=True
            )
            raise
    
    def _compute_ratio(self, metric_def: MetricDefinition, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Compute ratio-based metrics."""
        result = data.copy()
        
        # Parse formula (e.g., "revenue / spend")
        formula = metric_def.formula
        parts = [p.strip() for p in formula.split('/')]
        
        if len(parts) == 2:
            numerator = self._resolve_column(parts[0], data)
            denominator = self._resolve_column(parts[1], data)
            
            if numerator is not None and denominator is not None:
                # Avoid division by zero
                result[metric_def.metric_id] = numerator / denominator.replace(0, pd.NA)
            else:
                raise ValueError(f"Cannot resolve columns for {metric_def.metric_id}")
        
        return result
    
    def _compute_aggregation(self, metric_def: MetricDefinition, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Compute aggregation-based metrics."""
        # For aggregations, we typically return aggregated data
        group_by = kwargs.get('group_by', [])
        
        if group_by:
            grouped = data.groupby(group_by)
        else:
            # Single aggregation
            return pd.DataFrame([{
                metric_def.metric_id: self._evaluate_formula(metric_def.formula, data)
            }])
        
        # Apply aggregation
        # This is simplified - in production, parse formula more carefully
        aggregated = grouped.agg({
            col: 'sum' for col in data.select_dtypes(include=['number']).columns
        }).reset_index()
        
        return aggregated
    
    def _compute_delta(self, metric_def: MetricDefinition, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Compute delta-based metrics (MoM, WoW growth)."""
        # Parse formula (e.g., "(current - previous) / previous")
        # This requires time-series data
        time_col = kwargs.get('time_column', 'date')
        
        if time_col not in data.columns:
            raise ValueError(f"Time column {time_col} not found in data")
        
        # Sort by time
        data = data.sort_values(time_col)
        
        # Compute period-over-period
        result = data.copy()
        
        # Simplified - would need proper period grouping
        if len(data) > 1:
            result[f'{metric_def.metric_id}_delta'] = data.iloc[-1] - data.iloc[-2]
            result[f'{metric_def.metric_id}_pct_change'] = (
                (data.iloc[-1] - data.iloc[-2]) / data.iloc[-2].replace(0, pd.NA)
            ) * 100
        
        return result
    
    def _compute_simple(self, metric_def: MetricDefinition, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Compute simple metrics."""
        result = data.copy()
        value = self._evaluate_formula(metric_def.formula, data)
        result[metric_def.metric_id] = value
        return result
    
    def _resolve_column(self, column_name: str, data: pd.DataFrame) -> Optional[pd.Series]:
        """Resolve column name (handle aliases and variations)."""
        # Try exact match
        if column_name in data.columns:
            return data[column_name]
        
        # Try case-insensitive
        for col in data.columns:
            if col.lower() == column_name.lower():
                return data[col]
        
        # Try partial match
        for col in data.columns:
            if column_name.lower() in col.lower():
                return data[col]
        
        return None
    
    def _evaluate_formula(self, formula: str, data: pd.DataFrame) -> Any:
        """Evaluate formula expression."""
        # Simplified evaluation - in production, use proper expression parser
        if 'sum(' in formula.lower():
            # Extract column name
            col = formula.split('(')[1].split(')')[0].strip()
            col_series = self._resolve_column(col, data)
            if col_series is not None:
                return col_series.sum()
        
        # Default: try to evaluate as Python expression
        try:
            # Replace column names with actual values
            for col in data.columns:
                formula = formula.replace(col, f"data['{col}']")
            return eval(formula)
        except:
            return None
    
    def join(self, left_data: pd.DataFrame, right_data: pd.DataFrame, 
             join_keys: List[str], how: str = 'left') -> pd.DataFrame:
        """Join two dataframes with detailed logging."""
        logger.info(
            f"🔗 [COMPUTATION] Starting join operation | "
            f"left_rows={len(left_data)} | right_rows={len(right_data)} | "
            f"join_keys={join_keys} | how={how}"
        )
        
        # Normalize join keys
        left_keys = []
        right_keys = []
        
        for key_pair in join_keys:
            if isinstance(key_pair, dict):
                left_key = key_pair.get('left', key_pair.get('key'))
                right_key = key_pair.get('right', key_pair.get('key'))
            else:
                # Assume same column name
                left_key = key_pair
                right_key = key_pair
            
            # Validate keys exist
            if left_key not in left_data.columns:
                logger.warning(
                    f"⚠️ [COMPUTATION] Join key not found in left dataframe | "
                    f"key={left_key} | available_columns={list(left_data.columns)}"
                )
            if right_key not in right_data.columns:
                logger.warning(
                    f"⚠️ [COMPUTATION] Join key not found in right dataframe | "
                    f"key={right_key} | available_columns={list(right_data.columns)}"
                )
            
            left_keys.append(left_key)
            right_keys.append(right_key)
        
        # Log sample values for debugging
        if len(left_data) > 0:
            sample_left = left_data[left_keys].head(3).to_dict('records') if all(k in left_data.columns for k in left_keys) else []
            logger.debug(f"🔍 [COMPUTATION] Sample left join keys | sample={sample_left}")
        
        if len(right_data) > 0:
            sample_right = right_data[right_keys].head(3).to_dict('records') if all(k in right_data.columns for k in right_keys) else []
            logger.debug(f"🔍 [COMPUTATION] Sample right join keys | sample={sample_right}")
        
        # Perform join
        try:
            result = pd.merge(
                left_data,
                right_data,
                left_on=left_keys,
                right_on=right_keys,
                how=how,
                suffixes=('_left', '_right')
            )
            
            logger.info(
                f"{'✅' if len(result) > 0 else '⚠️'} [COMPUTATION] Join completed | "
                f"result_rows={len(result)} | "
                f"left_rows={len(left_data)} | right_rows={len(right_data)}"
            )
            
            if len(result) == 0 and len(left_data) > 0:
                logger.warning(
                    f"⚠️ [COMPUTATION] Join resulted in 0 rows! | "
                    f"This may indicate: 1) Join keys don't match, 2) Data type mismatch, "
                    f"3) No overlapping values | "
                    f"left_keys={left_keys} | right_keys={right_keys}"
                )
                # Log value counts for debugging
                if len(left_data) > 0:
                    for key in left_keys:
                        if key in left_data.columns:
                            unique_left = left_data[key].nunique()
                            logger.debug(f"📊 [COMPUTATION] Left key '{key}' has {unique_left} unique values")
                if len(right_data) > 0:
                    for key in right_keys:
                        if key in right_data.columns:
                            unique_right = right_data[key].nunique()
                            logger.debug(f"📊 [COMPUTATION] Right key '{key}' has {unique_right} unique values")
            
        except Exception as e:
            logger.error(
                f"❌ [COMPUTATION] Join failed | "
                f"error={str(e)} | join_keys={join_keys}",
                exc_info=True
            )
            raise
        
        return result
    
    def aggregate(self, data: pd.DataFrame, group_by: List[str], 
                  aggregations: Dict[str, str]) -> pd.DataFrame:
        """Aggregate data by groups."""
        import time
        start_time = time.time()
        
        logger.info(
            f"📥 [COMPUTATION] Input | "
            f"input_rows={len(data)} | "
            f"input_columns={list(data.columns)} | "
            f"group_by={group_by} | "
            f"aggregations={aggregations}"
        )
        
        logger.info(
            f"📊 [COMPUTATION] Aggregating data | "
            f"input_rows={len(data)} | group_by={group_by} | "
            f"aggregations={aggregations}"
        )
        
        agg_dict = {}
        for col, agg_func in aggregations.items():
            if col in data.columns:
                agg_dict[col] = agg_func
            else:
                logger.warning(
                    f"⚠️ [COMPUTATION] Aggregation column not found | "
                    f"column={col} | available_columns={list(data.columns)}"
                )
        
        if not agg_dict:
            logger.warning(f"⚠️ [COMPUTATION] No valid aggregation columns found, returning original data")
            return data
        
        # Validate group_by columns exist
        missing_group_cols = [col for col in group_by if col not in data.columns]
        if missing_group_cols:
            logger.warning(
                f"⚠️ [COMPUTATION] Group by columns not found | "
                f"missing={missing_group_cols} | available={list(data.columns)}"
            )
            # Remove missing columns from group_by
            group_by = [col for col in group_by if col in data.columns]
        
        if not group_by:
            # Single aggregation without grouping
            logger.debug(f"📊 [COMPUTATION] Performing single aggregation")
            result = pd.DataFrame([{
                col: getattr(data[col], agg_func)() if hasattr(data[col], agg_func) else data[col].agg(agg_func)
                for col, agg_func in agg_dict.items()
            }])
        else:
            grouped = data.groupby(group_by)
            result = grouped.agg(agg_dict).reset_index()
        
        logger.info(
            f"✅ [COMPUTATION] Aggregation completed | "
            f"result_rows={len(result)} | input_rows={len(data)}"
        )
        
        # Log outputs
        execution_time_ms = (time.time() - start_time) * 1000
        logger.info(
            f"📤 [COMPUTATION] Output | "
            f"result_rows={len(result)} | "
            f"result_columns={list(result.columns)} | "
            f"execution_time_ms={execution_time_ms:.2f}"
        )
        
        return result
    
    def top_n(self, data: pd.DataFrame, sort_by: List[str], limit: int, ascending: bool = False) -> pd.DataFrame:
        """Get top N rows sorted by specified columns."""
        import time
        start_time = time.time()
        
        logger.info(
            f"📥 [COMPUTATION] Input | "
            f"input_rows={len(data)} | "
            f"input_columns={list(data.columns)} | "
            f"sort_by={sort_by} | limit={limit} | ascending={ascending}"
        )
        
        logger.info(
            f"🔝 [COMPUTATION] Getting top N | "
            f"input_rows={len(data)} | sort_by={sort_by} | limit={limit} | ascending={ascending}"
        )
        
        if data.empty:
            logger.warning(f"⚠️ [COMPUTATION] Input data is empty for top_n operation")
            return data
        
        # Validate sort_by columns exist
        missing_cols = [col for col in sort_by if col not in data.columns]
        if missing_cols:
            logger.warning(
                f"⚠️ [COMPUTATION] Sort columns not found | "
                f"missing={missing_cols} | available={list(data.columns)}"
            )
            # Remove missing columns
            sort_by = [col for col in sort_by if col in data.columns]
        
        if not sort_by:
            logger.warning(f"⚠️ [COMPUTATION] No valid sort columns, returning first {limit} rows")
            return data.head(limit)
        
        # Sort and limit
        result = data.sort_values(by=sort_by, ascending=ascending).head(limit)
        
        logger.info(
            f"✅ [COMPUTATION] Top N completed | "
            f"result_rows={len(result)} | input_rows={len(data)}"
        )
        
        # Log outputs
        execution_time_ms = (time.time() - start_time) * 1000
        logger.info(
            f"📤 [COMPUTATION] Output | "
            f"result_rows={len(result)} | "
            f"result_columns={list(result.columns)} | "
            f"execution_time_ms={execution_time_ms:.2f}"
        )
        
        return result


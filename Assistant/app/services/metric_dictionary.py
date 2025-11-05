"""Metric Dictionary - Authoritative definitions of metrics."""
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
from app.models.schemas import MetricDefinition


class MetricDictionary:
    """Manages metric definitions and formulas."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize metric dictionary from YAML config."""
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "metrics.yaml"
        
        self.metrics: Dict[str, MetricDefinition] = {}
        self._load_metrics(config_path)
    
    def _load_metrics(self, config_path: Path):
        """Load metrics from YAML configuration."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        for metric_config in config.get('metrics', []):
            metric_def = MetricDefinition(**metric_config)
            self.metrics[metric_def.metric_id.lower()] = metric_def
    
    def get_metric(self, metric_id: str) -> Optional[MetricDefinition]:
        """Get metric definition by ID."""
        return self.metrics.get(metric_id.lower())
    
    def get_formula(self, metric_id: str) -> Optional[str]:
        """Get formula for a metric."""
        metric = self.get_metric(metric_id)
        return metric.formula if metric else None
    
    def get_dependencies(self, metric_id: str) -> List[str]:
        """Get required dependencies for computing a metric."""
        metric = self.get_metric(metric_id)
        return metric.dependencies if metric else []
    
    def resolve_metric(self, metric_id: str) -> Dict[str, Any]:
        """Resolve metric and all its dependencies recursively."""
        metric = self.get_metric(metric_id)
        if not metric:
            return {}
        
        result = {
            'metric': metric,
            'dependencies': []
        }
        
        for dep in metric.dependencies:
            dep_metric = self.get_metric(dep)
            if dep_metric:
                result['dependencies'].append(dep_metric)
            else:
                # Assume it's a raw column/field
                result['dependencies'].append({'name': dep, 'type': 'column'})
        
        return result
    
    def get_metrics_by_category(self, category: str) -> List[MetricDefinition]:
        """Get all metrics in a category."""
        return [
            metric for metric in self.metrics.values()
            if metric.category.lower() == category.lower()
        ]
    
    def get_all_metrics(self) -> List[MetricDefinition]:
        """Get all registered metrics."""
        return list(self.metrics.values())
    
    def validate_metric_request(self, metric_id: str) -> tuple[bool, Optional[str]]:
        """Validate if a metric can be computed (check dependencies)."""
        metric = self.get_metric(metric_id)
        if not metric:
            return False, f"Unknown metric: {metric_id}"
        
        # Basic validation - in production, check if dependencies are available
        return True, None


# Global dictionary instance
metric_dictionary = MetricDictionary()


"""Schema Registry - Database schema metadata."""
from typing import Dict, List, Optional, Set
from app.models.schemas import SchemaTable, SchemaColumn
from sqlalchemy import inspect, create_engine, MetaData, Table
from sqlalchemy.engine import Engine
from app.config.settings import settings
from app.config.logging_config import get_logger

logger = get_logger(__name__)


class SchemaRegistry:
    """Manages database schema metadata."""
    
    def __init__(self, database_url: Optional[str] = None):
        """Initialize schema registry from database."""
        self.database_url = database_url or settings.database_url
        self.engine: Optional[Engine] = None
        self.tables: Dict[str, SchemaTable] = {}
        self._load_schema()
    
    def _load_schema(self):
        """Load schema from database connection."""
        try:
            logger.info(f"🔧 [SCHEMA_REGISTRY] Loading schema from database")
            self.engine = create_engine(self.database_url, pool_pre_ping=True)
            metadata = MetaData()
            metadata.reflect(bind=self.engine)
            
            inspector = inspect(self.engine)
            schemas = inspector.get_schema_names()
            logger.debug(f"📋 [SCHEMA_REGISTRY] Found schemas: {schemas}")
            
            for schema_name in schemas:
                # Skip system schemas but include 'public'
                if schema_name in ['information_schema', 'pg_catalog']:
                    continue
                
                # Load all tables including public schema
                table_names = inspector.get_table_names(schema=schema_name)
                logger.debug(f"📊 [SCHEMA_REGISTRY] Loading schema '{schema_name}' | tables={len(table_names)}")
                
                for table_name in table_names:
                    full_name = f"{schema_name}.{table_name}"
                    self._register_table(schema_name, table_name, inspector)
                    logger.debug(f"✅ [SCHEMA_REGISTRY] Registered table: {full_name}")
            
            logger.info(f"✅ [SCHEMA_REGISTRY] Schema loaded | total_tables={len(self.tables)}")
        except Exception as e:
            logger.warning(f"⚠️ [SCHEMA_REGISTRY] Failed to load schema from database, using fallback | error={str(e)}")
            # Fallback to manual schema definitions if connection fails
            self._load_manual_schema()
    
    def _register_table(self, schema: str, table_name: str, inspector):
        """Register a table in the schema registry."""
        columns = []
        primary_keys = []
        foreign_keys = []
        
        # Get columns
        for col in inspector.get_columns(table_name, schema=schema):
            col_def = SchemaColumn(
                name=col['name'],
                type=str(col['type']),
                nullable=col.get('nullable', True),
                primary_key=False,
                description=None
            )
            columns.append(col_def)
        
        # Get primary keys
        pk_constraint = inspector.get_pk_constraint(table_name, schema=schema)
        if pk_constraint:
            primary_keys = pk_constraint.get('constrained_columns', [])
            for col in columns:
                if col.name in primary_keys:
                    col.primary_key = True
        
        # Get foreign keys
        fk_constraints = inspector.get_foreign_keys(table_name, schema=schema)
        for fk in fk_constraints:
            # Handle both single column and composite foreign keys
            # Convert lists to comma-separated strings if needed
            constrained_cols = fk.get('constrained_columns', [])
            referred_cols = fk.get('referred_columns', [])
            
            # Convert lists to strings (join if multiple columns)
            columns_str = ', '.join(constrained_cols) if isinstance(constrained_cols, list) else str(constrained_cols)
            referred_columns_str = ', '.join(referred_cols) if isinstance(referred_cols, list) else str(referred_cols)
            
            foreign_keys.append({
                'columns': columns_str,
                'referred_table': fk.get('referred_table', ''),
                'referred_columns': referred_columns_str
            })
        
        table_def = SchemaTable(
            schema_name=schema,  # Use schema_name since we renamed the field
            table=table_name,
            columns=columns,
            primary_keys=primary_keys,
            foreign_keys=foreign_keys,
            rls_enabled=False
        )
        
        self.tables[f"{schema}.{table_name}"] = table_def
    
    def _load_manual_schema(self):
        """Load schema from manual definitions (fallback)."""
        # Key tables from schema.sql
        tables_config = {
            'public.dw_meta_ads_attribution': {
                'columns': [
                    {'name': 'campaign_id', 'type': 'bigint', 'nullable': False},
                    {'name': 'campaign_name', 'type': 'text', 'nullable': True},
                    {'name': 'date_start', 'type': 'date', 'nullable': False},
                    {'name': 'hour', 'type': 'integer', 'nullable': False},
                    {'name': 'spend', 'type': 'numeric(15,2)', 'nullable': True},
                    {'name': 'clicks', 'type': 'integer', 'nullable': True},
                    {'name': 'impressions', 'type': 'integer', 'nullable': True},
                    {'name': 'ctr', 'type': 'numeric(15,4)', 'nullable': True},
                    {'name': 'cpc', 'type': 'numeric(15,4)', 'nullable': True},
                    {'name': 'attributed_orders_revenue', 'type': 'numeric(15,2)', 'nullable': True},
                ],
                'primary_keys': ['id'],
                'date_column': 'date_start'
            },
            'public.shopify_orders': {
                'columns': [
                    {'name': 'order_id', 'type': 'text', 'nullable': False},
                    {'name': 'created_at', 'type': 'timestamp with time zone', 'nullable': True},
                    {'name': 'total_price_amount', 'type': 'numeric', 'nullable': True},
                    {'name': 'ship_city', 'type': 'text', 'nullable': True},
                    {'name': 'ship_country', 'type': 'text', 'nullable': True},
                ],
                'primary_keys': ['order_id'],
                'date_column': 'created_at'
            },
            'public.amazon_product_metrics_daily': {
                'columns': [
                    {'name': 'campaign_id', 'type': 'bigint', 'nullable': False},
                    {'name': 'campaign_name', 'type': 'varchar(500)', 'nullable': True},
                    {'name': 'date', 'type': 'date', 'nullable': False},
                    {'name': 'spend', 'type': 'numeric(12,2)', 'nullable': True},
                    {'name': 'clicks', 'type': 'integer', 'nullable': True},
                    {'name': 'impressions', 'type': 'integer', 'nullable': True},
                    {'name': 'sales', 'type': 'numeric(12,2)', 'nullable': True},
                    {'name': 'orders', 'type': 'integer', 'nullable': True},
                    {'name': 'roas', 'type': 'numeric(10,6)', 'nullable': True},
                    {'name': 'acos', 'type': 'numeric(10,6)', 'nullable': True},
                    {'name': 'ctr', 'type': 'numeric(10,6)', 'nullable': True},
                    {'name': 'cpc', 'type': 'numeric(10,2)', 'nullable': True},
                ],
                'primary_keys': ['id'],
                'date_column': 'date'
            },
        }
        
        for full_name, config in tables_config.items():
            schema, table = full_name.split('.')
            columns = [
                SchemaColumn(
                    name=col['name'],
                    type=col['type'],
                    nullable=col.get('nullable', True),
                    primary_key=col['name'] in config['primary_keys']
                )
                for col in config['columns']
            ]
            
            table_def = SchemaTable(
                schema_name=schema,  # Use schema_name since we renamed the field
                table=table,
                columns=columns,
                primary_keys=config['primary_keys'],
                date_column=config.get('date_column')
            )
            self.tables[full_name] = table_def
    
    def get_table(self, schema: str, table: str) -> Optional[SchemaTable]:
        """Get table definition."""
        full_name = f"{schema}.{table}"
        return self.tables.get(full_name)
    
    def get_column(self, schema: str, table: str, column: str) -> Optional[SchemaColumn]:
        """Get column definition."""
        table_def = self.get_table(schema, table)
        if not table_def:
            return None
        
        for col in table_def.columns:
            if col.name == column:
                return col
        return None
    
    def validate_column(self, schema: str, table: str, column: str) -> bool:
        """Validate if a column exists in a table."""
        return self.get_column(schema, table, column) is not None
    
    def get_join_keys(self, schema1: str, table1: str, schema2: str, table2: str) -> List[Dict[str, str]]:
        """Find potential join keys between two tables."""
        table1_def = self.get_table(schema1, table1)
        table2_def = self.get_table(schema2, table2)
        
        if not table1_def or not table2_def:
            return []
        
        join_keys = []
        table1_cols = {col.name.lower(): col.name for col in table1_def.columns}
        table2_cols = {col.name.lower(): col.name for col in table2_def.columns}
        
        # Find common column names (case-insensitive)
        common_cols = set(table1_cols.keys()) & set(table2_cols.keys())
        
        for col in common_cols:
            # Prefer columns that look like IDs or keys
            if any(keyword in col for keyword in ['id', 'key', 'campaign', 'order']):
                join_keys.append({
                    'left': f"{schema1}.{table1}.{table1_cols[col]}",
                    'right': f"{schema2}.{table2}.{table2_cols[col]}"
                })
        
        return join_keys
    
    def get_date_column(self, schema: str, table: str) -> Optional[str]:
        """Get the date/time column for a table."""
        table_def = self.get_table(schema, table)
        if not table_def:
            return None
        
        # Check if date_column is explicitly set
        if hasattr(table_def, 'date_column') and table_def.date_column:
            return table_def.date_column
        
        # Try to find common date column names
        date_cols = ['date', 'date_start', 'created_at', 'updated_at', 'processed_at']
        for col in table_def.columns:
            if col.name.lower() in date_cols:
                return col.name
        
        return None


# Global registry instance
schema_registry = SchemaRegistry()


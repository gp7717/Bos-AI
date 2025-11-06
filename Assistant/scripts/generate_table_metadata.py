"""Script to generate comprehensive table_metadata.yaml from database schema."""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.schema_registry import schema_registry
from app.config.logging_config import get_logger
import yaml
from typing import Dict, List, Any

logger = get_logger(__name__)


def categorize_table(full_name: str) -> str:
    """Categorize table by name patterns."""
    name_lower = full_name.lower()
    
    if 'shopify' in name_lower:
        return 'shopify'
    elif 'meta' in name_lower or 'facebook' in name_lower or 'fb' in name_lower:
        return 'meta_ads'
    elif 'google' in name_lower or 'gads' in name_lower:
        return 'google_ads'
    elif 'amazon' in name_lower:
        return 'amazon_ads'
    elif 'dw_' in name_lower:
        return 'data_warehouse'
    elif 'customer' in name_lower:
        return 'customer'
    elif 'product' in name_lower:
        return 'product'
    elif 'order' in name_lower:
        return 'order'
    else:
        return 'other'


def infer_date_column(columns: List[Any]) -> str:
    """Infer date column from column names."""
    date_keywords = ['created_at', 'date_start', 'date', 'updated_at', 'processed_at', 'order_date']
    for col in columns:
        if col.name.lower() in date_keywords:
            return col.name
    return None


def infer_purpose(full_name: str, table_name: str, columns: List[Any]) -> str:
    """Infer table purpose from name and columns."""
    name_lower = full_name.lower()
    col_names = [col.name.lower() for col in columns]
    
    if 'order' in name_lower and 'line_item' in name_lower:
        return "Order line items - individual products in each order"
    elif 'order' in name_lower:
        return "Orders table - contains order-level information"
    elif 'product' in name_lower and 'variant' in name_lower:
        return "Product variants - contains product-level information including SKU, product_id, and product_title"
    elif 'product' in name_lower:
        return "Products table - contains product information"
    elif 'meta' in name_lower or 'facebook' in name_lower:
        return "Meta (Facebook) Ads attribution data"
    elif 'google' in name_lower:
        return "Google Ads attribution data"
    elif 'amazon' in name_lower:
        return "Amazon Ads data"
    elif 'customer' in name_lower:
        return "Customer information"
    elif 'attribution' in name_lower:
        return "Marketing attribution data"
    else:
        return f"Table: {full_name}"


def generate_table_metadata() -> Dict[str, Any]:
    """Generate comprehensive table metadata from schema registry."""
    logger.info("🔧 Generating table metadata from database schema...")
    
    # Get all tables from registry
    all_tables = schema_registry.tables
    logger.info(f"📊 Found {len(all_tables)} tables in database")
    
    # Categorize tables
    categorized = {}
    for full_name, table_def in all_tables.items():
        category = categorize_table(full_name)
        if category not in categorized:
            categorized[category] = []
        categorized[category].append((full_name, table_def))
    
    logger.info(f"📋 Categorized into {len(categorized)} categories: {list(categorized.keys())}")
    
    # Generate metadata structure
    metadata = {
        'tables': {},
        'common_patterns': {},
        'critical_rules': []
    }
    
    # Process each table
    for category, tables in categorized.items():
        logger.info(f"📝 Processing {len(tables)} tables in category: {category}")
        
        for full_name, table_def in tables:
            schema, table_name = full_name.split('.', 1)
            
            # Infer metadata
            purpose = infer_purpose(full_name, table_name, table_def.columns)
            date_col = infer_date_column(table_def.columns) or schema_registry.get_date_column(schema, table_name)
            
            # Build all_columns list
            all_columns = []
            for col in table_def.columns:
                col_meta = {
                    'name': col.name,
                    'type': str(col.type),
                    'nullable': col.nullable,
                    'primary_key': col.primary_key,
                    'purpose': f"Column: {col.name}"
                }
                
                # Add foreign key info if available
                for fk in table_def.foreign_keys:
                    fk_cols = fk.get('columns', '')
                    if isinstance(fk_cols, str):
                        fk_cols = [c.strip() for c in fk_cols.split(',')]
                    if col.name in fk_cols:
                        col_meta['foreign_key'] = True
                        col_meta['references'] = f"{fk.get('referred_table', '')}.{fk.get('referred_columns', '')}"
                
                if date_col and col.name == date_col:
                    col_meta['date_column'] = True
                
                all_columns.append(col_meta)
            
            # Build key_columns (primary keys, foreign keys, date columns, and important columns)
            key_columns = []
            for col in table_def.columns:
                if (col.primary_key or 
                    any(col.name in fk.get('columns', '') if isinstance(fk.get('columns', ''), str) else col.name in fk.get('columns', []) 
                        for fk in table_def.foreign_keys) or
                    (date_col and col.name == date_col) or
                    any(keyword in col.name.lower() for keyword in ['id', 'name', 'title', 'amount', 'revenue', 'spend', 'cost'])):
                    
                    col_meta = {
                        'name': col.name,
                        'type': str(col.type),
                        'primary_key': col.primary_key,
                        'purpose': f"Column: {col.name}"
                    }
                    
                    if col.primary_key:
                        col_meta['purpose'] = f"Primary key - unique identifier"
                    if date_col and col.name == date_col:
                        col_meta['date_column'] = True
                        col_meta['purpose'] = f"Date column for filtering - use this for date range queries"
                    
                    for fk in table_def.foreign_keys:
                        fk_cols = fk.get('columns', '')
                        if isinstance(fk_cols, str):
                            fk_cols = [c.strip() for c in fk_cols.split(',')]
                        if col.name in fk_cols:
                            col_meta['foreign_key'] = True
                            col_meta['references'] = f"{fk.get('referred_table', '')}.{fk.get('referred_columns', '')}"
                            col_meta['purpose'] = f"Foreign key → {col_meta['references']}"
                    
                    key_columns.append(col_meta)
            
            # Build relationships
            relationships = []
            for fk in table_def.foreign_keys:
                fk_cols = fk.get('columns', '')
                if isinstance(fk_cols, str):
                    fk_cols = [c.strip() for c in fk_cols.split(',')]
                ref_cols = fk.get('referred_columns', '')
                if isinstance(ref_cols, str):
                    ref_cols = [c.strip() for c in ref_cols.split(',')]
                
                relationships.append({
                    'type': 'many_to_one',
                    'local_column': fk_cols[0] if fk_cols else '',
                    'foreign_table': fk.get('referred_table', ''),
                    'foreign_column': ref_cols[0] if ref_cols else '',
                    'purpose': f"Links to {fk.get('referred_table', '')}",
                    'join_example': f"JOIN {fk.get('referred_table', '')} ON {full_name}.{fk_cols[0] if fk_cols else ''} = {fk.get('referred_table', '')}.{ref_cols[0] if ref_cols else ''}"
                })
            
            # Build table metadata
            table_metadata = {
                'purpose': purpose,
                'all_columns': all_columns,
                'key_columns': key_columns,
                'relationships': relationships,
                'quirks': [],
                'json_columns': []
            }
            
            if date_col:
                table_metadata['date_column'] = date_col
                table_metadata['quirks'].append(f"Date filtering: WHERE {date_col} BETWEEN :date_start AND :date_end")
            
            # Add category-specific quirks
            if category == 'shopify' and 'order_line_items' in full_name:
                table_metadata['quirks'].append("⚠️ CRITICAL: This table does NOT have 'sku' column - it's in shopify_product_variants")
                table_metadata['quirks'].append("⚠️ CRITICAL: This table does NOT have 'product_id' column - it's in shopify_product_variants")
                table_metadata['columns_not_in_table'] = [
                    "⚠️ CRITICAL: sku - DOES NOT EXIST HERE! Use shopify_product_variants.sku instead",
                    "⚠️ CRITICAL: product_id - DOES NOT EXIST HERE! Use shopify_product_variants.product_id instead"
                ]
            
            metadata['tables'][full_name] = table_metadata
    
    logger.info(f"✅ Generated metadata for {len(metadata['tables'])} tables")
    return metadata


def main():
    """Main function to generate and save table metadata."""
    try:
        metadata = generate_table_metadata()
        
        # Load existing metadata to preserve common_patterns and critical_rules
        config_path = Path(__file__).parent.parent / "config" / "table_metadata.yaml"
        if config_path.exists():
            with open(config_path, 'r') as f:
                existing = yaml.safe_load(f) or {}
                if 'common_patterns' in existing:
                    metadata['common_patterns'] = existing['common_patterns']
                if 'critical_rules' in existing:
                    metadata['critical_rules'] = existing['critical_rules']
        
        # Save to file
        with open(config_path, 'w') as f:
            yaml.dump(metadata, f, default_flow_style=False, sort_keys=False, allow_unicode=True, width=120)
        
        logger.info(f"✅ Saved table metadata to {config_path}")
        logger.info(f"📊 Total tables: {len(metadata['tables'])}")
        
        # Print summary
        print("\n" + "="*80)
        print("TABLE METADATA GENERATION SUMMARY")
        print("="*80)
        print(f"Total tables processed: {len(metadata['tables'])}")
        print("\nTables by category:")
        categories = {}
        for full_name in metadata['tables'].keys():
            cat = categorize_table(full_name)
            categories[cat] = categories.get(cat, 0) + 1
        for cat, count in sorted(categories.items()):
            print(f"  {cat}: {count} tables")
        print("\n✅ Metadata saved successfully!")
        
    except Exception as e:
        logger.error(f"❌ Failed to generate table metadata | error={str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()


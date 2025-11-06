"""Script to generate table_metadata.yaml from database schema."""
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.schema_registry import schema_registry
from app.config.logging_config import get_logger

logger = get_logger(__name__)

def parse_schema_from_database() -> Dict[str, Any]:
    """Parse schema from database using schema_registry."""
    tables = {}
    
    # Get all tables from schema registry
    for full_name, table_def in schema_registry.tables.items():
        schema, table_name = full_name.split('.', 1) if '.' in full_name else ('public', full_name)
        
        # Build all_columns list
        all_columns = []
        for col in table_def.columns:
            col_info = {
                'name': col.name,
                'type': str(col.type),
                'nullable': col.nullable,
                'primary_key': col.primary_key,
                'purpose': infer_column_purpose(col.name, str(col.type), table_name)
            }
            
            # Check if it's a date column
            if any(dt in str(col.type).upper() for dt in ['DATE', 'TIMESTAMP', 'TIME']):
                col_info['date_column'] = True
            
            # Check if it's JSON
            if 'JSON' in str(col.type).upper():
                col_info['json_column'] = True
            
            # Add foreign key info (but don't override primary key purpose)
            for fk in table_def.foreign_keys:
                fk_cols = fk.get('columns', '')
                if isinstance(fk_cols, str):
                    fk_cols = [c.strip() for c in fk_cols.split(',')]
                if col.name in fk_cols and not col.primary_key:
                    col_info['foreign_key'] = True
                    col_info['references'] = f"{fk.get('referred_table', '')}.{fk.get('referred_columns', '')}"
                    col_info['purpose'] = f"Foreign key → {col_info['references']}"
            
            all_columns.append(col_info)
        
        # Build key_columns
        key_columns = []
        for col in table_def.columns:
            if (col.primary_key or 
                any(keyword in col.name.lower() for keyword in ['id', 'name', 'title', 'amount', 'revenue', 'spend', 'cost', 'date', 'sku', 'product']) or
                any(dt in str(col.type).upper() for dt in ['DATE', 'TIMESTAMP'])):
                key_col = {
                    'name': col.name,
                    'type': str(col.type),
                    'primary_key': col.primary_key,
                    'purpose': infer_column_purpose(col.name, str(col.type), table_name)
                }
                if any(dt in str(col.type).upper() for dt in ['DATE', 'TIMESTAMP']):
                    key_col['date_column'] = True
                key_columns.append(key_col)
        
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
        
        # Infer date column
        date_column = schema_registry.get_date_column(schema, table_name)
        
        # Infer purpose
        purpose = infer_table_purpose(table_name, all_columns)
        
        # Identify JSON columns
        json_columns = []
        for col in all_columns:
            if col.get('json_column'):
                json_columns.append({
                    'name': col['name'],
                    'purpose': f"JSONB column: {col['purpose']}",
                    'extraction_example': f"jsonb_array_elements({col['name']}) AS {col['name']}_data"
                })
        
        # Identify columns NOT in this table
        columns_not_in_table = infer_columns_not_in_table(table_name, all_columns)
        
        tables[full_name] = {
            'purpose': purpose,
            'alias_suggestions': generate_alias_suggestions(table_name),
            'all_columns': all_columns,
            'key_columns': key_columns,
            'relationships': relationships,
            'columns_not_in_table': columns_not_in_table,
            'quirks': infer_quirks(table_name, all_columns, relationships),
            'json_columns': json_columns
        }
        
        if date_column:
            tables[full_name]['date_column'] = date_column
    
    return tables

def infer_column_purpose(col_name: str, col_type: str, table_name: str) -> str:
    """Infer column purpose from name and context."""
    col_lower = col_name.lower()
    table_lower = table_name.lower()
    
    # Primary key detection
    if 'id' in col_lower:
        # Check if this is likely the primary key of this table
        if (col_lower == 'order_id' and 'shopify_orders' in table_lower) or \
           (col_lower == 'item_id' and 'shopify_order_line_items' in table_lower) or \
           (col_lower == 'variant_id' and 'shopify_product_variants' in table_lower):
            return "Primary key - unique identifier"
        elif 'order_id' in col_lower and 'shopify_order_line_items' in table_lower:
            return f"Foreign key → shopify_orders.order_id (links to parent order)"
        elif 'variant_id' in col_lower and 'shopify_order_line_items' in table_lower:
            return f"Foreign key → shopify_product_variants.variant_id (links to product variant)"
        else:
            return f"Identifier: {col_name}"
    
    if 'sku' in col_lower:
        return "Stock keeping unit identifier - THIS IS WHERE SKU IS!"
    if 'product_id' in col_lower:
        return "Product identifier - groups variants of the same product"
    if 'product_title' in col_lower:
        return "Product name/title"
    if 'created_at' in col_lower or 'date_start' in col_lower:
        return "Date column for filtering - use this for date range queries"
    if 'revenue' in col_lower or 'amount' in col_lower:
        return "Monetary amount"
    if 'quantity' in col_lower:
        return "Quantity of items"
    if 'spend' in col_lower or 'cost' in col_lower:
        return "Cost/spend amount"
    
    return f"Column: {col_name}"

def infer_table_purpose(table_name: str, columns: List[Dict]) -> str:
    """Infer table purpose from name and columns."""
    name_lower = table_name.lower()
    
    if 'shopify_orders' in name_lower:
        return "Main orders table - contains order-level information and totals"
    if 'shopify_order_line_items' in name_lower:
        return "Order line items - individual products in each order"
    if 'shopify_product_variants' in name_lower:
        return "Product variants - contains product-level information including SKU, product_id, and product_title"
    if 'meta_ads' in name_lower or 'dw_meta' in name_lower:
        return "Meta (Facebook) Ads attribution data - hourly granularity"
    if 'google_ads' in name_lower or 'dw_google' in name_lower:
        return "Google Ads attribution data - hourly granularity"
    if 'organic' in name_lower:
        return "Organic traffic attribution data"
    if 'amazon' in name_lower:
        return "Amazon Ads data"
    
    return f"Table: {table_name}"

def infer_relationships(table_name: str, columns: List[Dict], existing_tables: Dict) -> List[Dict]:
    """Infer relationships from column names."""
    relationships = []
    name_lower = table_name.lower()
    
    # shopify_orders -> shopify_order_line_items
    if 'shopify_orders' in name_lower:
        for col in columns:
            if col['name'] == 'order_id':
                relationships.append({
                    'type': 'one_to_many',
                    'local_column': 'order_id',
                    'foreign_table': 'public.shopify_order_line_items',
                    'foreign_column': 'order_id',
                    'purpose': 'Each order can have multiple line items',
                    'join_example': 'JOIN public.shopify_order_line_items oli ON o.order_id = oli.order_id'
                })
    
    # shopify_order_line_items -> shopify_orders
    if 'shopify_order_line_items' in name_lower:
        for col in columns:
            if col['name'] == 'order_id':
                relationships.append({
                    'type': 'many_to_one',
                    'local_column': 'order_id',
                    'foreign_table': 'public.shopify_orders',
                    'foreign_column': 'order_id',
                    'purpose': 'Many line items belong to one order',
                    'join_example': 'FROM public.shopify_order_line_items oli JOIN public.shopify_orders o ON oli.order_id = o.order_id'
                })
            elif col['name'] == 'variant_id':
                relationships.append({
                    'type': 'many_to_one',
                    'local_column': 'variant_id',
                    'foreign_table': 'public.shopify_product_variants',
                    'foreign_column': 'variant_id',
                    'purpose': 'Many line items can reference the same product variant',
                    'join_example': 'JOIN public.shopify_product_variants pv ON oli.variant_id = pv.variant_id'
                })
    
    # shopify_product_variants -> shopify_order_line_items
    if 'shopify_product_variants' in name_lower:
        for col in columns:
            if col['name'] == 'variant_id':
                relationships.append({
                    'type': 'one_to_many',
                    'local_column': 'variant_id',
                    'foreign_table': 'public.shopify_order_line_items',
                    'foreign_column': 'variant_id',
                    'purpose': 'One variant can appear in many order line items',
                    'join_example': 'FROM public.shopify_product_variants pv JOIN public.shopify_order_line_items oli ON pv.variant_id = oli.variant_id'
                })
    
    return relationships

def infer_columns_not_in_table(table_name: str, columns: List[Dict]) -> List[str]:
    """Identify columns that are commonly mistaken to be in this table."""
    name_lower = table_name.lower()
    column_names = {col['name'].lower() for col in columns}
    
    not_in_table = []
    
    if 'shopify_order_line_items' in name_lower:
        if 'sku' not in column_names:
            not_in_table.append("⚠️ CRITICAL: sku - DOES NOT EXIST HERE! Use shopify_product_variants.sku instead")
        if 'product_id' not in column_names:
            not_in_table.append("⚠️ CRITICAL: product_id - DOES NOT EXIST HERE! Use shopify_product_variants.product_id instead")
        if 'product_title' not in column_names:
            not_in_table.append("⚠️ CRITICAL: product_title - DOES NOT EXIST HERE! Use shopify_product_variants.product_title instead")
    
    if 'shopify_orders' in name_lower:
        if 'sku' not in column_names:
            not_in_table.append("sku (use shopify_product_variants.sku instead)")
        if 'product_id' not in column_names:
            not_in_table.append("product_id (use shopify_product_variants.product_id instead)")
        if 'variant_id' not in column_names:
            not_in_table.append("variant_id (use shopify_order_line_items.variant_id instead)")
        if 'quantity' not in column_names:
            not_in_table.append("quantity (use shopify_order_line_items.quantity instead)")
    
    return not_in_table

def infer_quirks(table_name: str, columns: List[Dict], relationships: List[Dict]) -> List[str]:
    """Infer table-specific quirks and important notes."""
    quirks = []
    name_lower = table_name.lower()
    
    if 'shopify_orders' in name_lower:
        quirks.append("⚠️ CRITICAL: DO NOT use 'order_date' - use 'created_at' instead")
        quirks.append("Date filtering: WHERE created_at BETWEEN :date_start AND :date_end")
        quirks.append("This table contains ORDER-LEVEL data only, not product/line-item details")
        quirks.append("To get product details, you MUST JOIN with shopify_order_line_items and shopify_product_variants")
    
    if 'shopify_order_line_items' in name_lower:
        quirks.append("⚠️ CRITICAL: This table does NOT have 'sku' column - it's in shopify_product_variants")
        quirks.append("⚠️ CRITICAL: This table does NOT have 'product_id' column - it's in shopify_product_variants")
        quirks.append("Revenue calculation: SUM(oli.quantity * oli.discounted_unit_price_amount)")
        quirks.append("Always use variant_id to link to product_variants table")
    
    if 'shopify_product_variants' in name_lower:
        quirks.append("⚠️ IMPORTANT: sku, product_id, and product_title are in THIS table, NOT in shopify_order_line_items")
        quirks.append("To access sku from orders: JOIN public.shopify_order_line_items oli ON o.order_id = oli.order_id JOIN public.shopify_product_variants pv ON oli.variant_id = pv.variant_id, then use pv.sku")
    
    if 'meta_ads' in name_lower or 'dw_meta' in name_lower:
        date_col = next((col['name'] for col in columns if 'date' in col['name'].lower()), None)
        if date_col:
            quirks.append(f"Date filtering: WHERE {date_col} BETWEEN :date_start AND :date_end")
        quirks.append("Has hourly granularity - aggregate by date if daily data needed")
        quirks.append("Use SUM() and GROUP BY date_start to get daily aggregates")
    
    if 'google_ads' in name_lower or 'dw_google' in name_lower:
        quirks.append("⚠️ Uses 'cost_amount' instead of 'spend' (different from Meta Ads)")
        date_col = next((col['name'] for col in columns if 'date' in col['name'].lower()), None)
        if date_col:
            quirks.append(f"Date filtering: WHERE {date_col} BETWEEN :date_start AND :date_end")
        quirks.append("Has hourly granularity - aggregate by date if daily data needed")
    
    return quirks

def generate_alias_suggestions(table_name: str) -> List[str]:
    """Generate alias suggestions for table."""
    name_lower = table_name.lower().replace('public.', '')
    
    if 'shopify_orders' in name_lower:
        return ['o', 'ord']
    if 'shopify_order_line_items' in name_lower:
        return ['oli', 'line_items']
    if 'shopify_product_variants' in name_lower:
        return ['pv', 'variants']
    if 'meta_ads' in name_lower or 'dw_meta' in name_lower:
        return ['meta', 'fb_ads']
    if 'google_ads' in name_lower or 'dw_google' in name_lower:
        return ['google', 'gads']
    
    # Generate from name
    parts = name_lower.split('_')
    if len(parts) > 1:
        return [''.join(p[0] for p in parts), parts[-1]]
    return [name_lower[:3]]

def generate_common_patterns() -> Dict[str, Any]:
    """Generate common query patterns."""
    return {
        'get_sku_from_orders': {
            'description': 'To get SKU from orders, you MUST join through line_items to product_variants',
            'sql': '''SELECT 
  o.order_id,
  o.created_at,
  pv.sku,
  pv.product_title,
  oli.quantity,
  oli.discounted_unit_price_amount
FROM public.shopify_orders o
JOIN public.shopify_order_line_items oli ON o.order_id = oli.order_id
JOIN public.shopify_product_variants pv ON oli.variant_id = pv.variant_id
WHERE o.created_at BETWEEN :date_start AND :date_end''',
            'notes': [
                '⚠️ You CANNOT use oli.sku - it doesn\'t exist!',
                '⚠️ You MUST join shopify_product_variants to get sku',
                'Always use: pv.sku (not oli.sku)'
            ]
        },
        'get_product_id_from_orders': {
            'description': 'To get product_id from orders, join through line_items to product_variants',
            'sql': '''SELECT 
  o.order_id,
  pv.product_id,
  pv.product_title,
  SUM(oli.quantity * oli.discounted_unit_price_amount) AS revenue
FROM public.shopify_orders o
JOIN public.shopify_order_line_items oli ON o.order_id = oli.order_id
JOIN public.shopify_product_variants pv ON oli.variant_id = pv.variant_id
WHERE o.created_at BETWEEN :date_start AND :date_end
GROUP BY o.order_id, pv.product_id, pv.product_title''',
            'notes': [
                '⚠️ product_id is in shopify_product_variants, NOT in shopify_order_line_items',
                'Always use: pv.product_id (not oli.product_id)'
            ]
        },
        'top_n_skus_by_revenue': {
            'description': 'Get top N SKUs by revenue - requires joining all three tables',
            'sql': '''SELECT 
  pv.sku,
  pv.product_title,
  SUM(oli.quantity * oli.discounted_unit_price_amount) AS revenue
FROM public.shopify_orders o
JOIN public.shopify_order_line_items oli ON o.order_id = oli.order_id
JOIN public.shopify_product_variants pv ON oli.variant_id = pv.variant_id
WHERE o.created_at BETWEEN :date_start AND :date_end
GROUP BY pv.sku, pv.product_title
ORDER BY revenue DESC
LIMIT :top_n''',
            'notes': [
                '⚠️ sku is in pv (product_variants), NOT in oli (order_line_items)',
                '⚠️ You MUST join all three tables: orders → line_items → product_variants'
            ]
        }
    }

def generate_critical_rules() -> List[Dict[str, str]]:
    """Generate critical rules for SQL generation."""
    return [
        {
            'rule': 'Column Location Rules',
            'description': '''- sku → ONLY in shopify_product_variants (pv.sku)
- product_id → ONLY in shopify_product_variants (pv.product_id)
- product_title → ONLY in shopify_product_variants (pv.product_title)
- variant_id → in shopify_order_line_items (oli.variant_id)
- quantity → in shopify_order_line_items (oli.quantity)
- order_id → in both shopify_orders (o.order_id) and shopify_order_line_items (oli.order_id)
- created_at → in shopify_orders (o.created_at) for date filtering'''
        },
        {
            'rule': 'Required JOINs',
            'description': '''To access sku, product_id, or product_title from orders:
1. Start with shopify_orders (o)
2. JOIN shopify_order_line_items (oli) ON o.order_id = oli.order_id
3. JOIN shopify_product_variants (pv) ON oli.variant_id = pv.variant_id
4. Then use pv.sku, pv.product_id, pv.product_title'''
        },
        {
            'rule': 'Common Mistakes to Avoid',
            'description': '''- ❌ DO NOT use: oli.sku (doesn't exist)
- ❌ DO NOT use: oli.product_id (doesn't exist)
- ❌ DO NOT use: oli.product_title (doesn't exist)
- ✅ DO use: pv.sku (after joining product_variants)
- ✅ DO use: pv.product_id (after joining product_variants)
- ✅ DO use: pv.product_title (after joining product_variants)'''
        }
    ]

def main():
    """Main function to generate table metadata from database."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    output_path = project_root / "config" / "table_metadata.yaml"
    
    print(f"📖 Reading schema from database...")
    tables = parse_schema_from_database()
    
    print(f"📊 Parsed {len(tables)} tables from schema")
    
    # Generate metadata structure
    metadata = {
        'tables': tables,
        'common_patterns': generate_common_patterns(),
        'critical_rules': generate_critical_rules()
    }
    
    # Write to YAML file
    print(f"💾 Writing metadata to: {output_path}")
    with open(output_path, 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False, allow_unicode=True, width=120)
    
    print(f"✅ Generated metadata for {len(tables)} tables")
    print(f"📋 Common patterns: {len(metadata['common_patterns'])}")
    print(f"📋 Critical rules: {len(metadata['critical_rules'])}")
    
    # Print summary
    print("\n" + "="*80)
    print("TABLE METADATA GENERATION SUMMARY")
    print("="*80)
    print(f"Total tables: {len(tables)}")
    print("\nTables:")
    for table_name in sorted(tables.keys()):
        table_info = tables[table_name]
        print(f"  {table_name}:")
        print(f"    - Columns: {len(table_info['all_columns'])}")
        print(f"    - Key columns: {len(table_info['key_columns'])}")
        print(f"    - Relationships: {len(table_info['relationships'])}")
        if table_info.get('date_column'):
            print(f"    - Date column: {table_info['date_column']}")
    print("\n✅ Metadata saved successfully!")

if __name__ == "__main__":
    main()


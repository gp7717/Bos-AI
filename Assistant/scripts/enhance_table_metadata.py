"""Script to enhance table metadata with detailed descriptions for key tables."""
import sys
from pathlib import Path
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config.logging_config import get_logger

logger = get_logger(__name__)


# Enhanced metadata for key tables
KEY_TABLE_ENHANCEMENTS = {
    'public.shopify_orders': {
        'purpose': 'Main orders table - contains order-level information and totals',
        'alias_suggestions': ['o', 'ord'],
        'enhanced_key_columns': [
            {
                'name': 'order_id',
                'purpose': 'Primary key - unique identifier for each order',
                'type': 'text',
                'primary_key': True
            },
            {
                'name': 'created_at',
                'purpose': 'Order creation timestamp - USE THIS for date filtering, NOT \'order_date\'',
                'type': 'timestamp with time zone',
                'date_column': True
            },
            {
                'name': 'total_price_amount',
                'purpose': 'Order total revenue amount (order-level, not line-item level)',
                'type': 'numeric'
            }
        ],
        'columns_not_in_table': [
            'sku (use shopify_product_variants.sku instead)',
            'product_id (use shopify_product_variants.product_id instead)',
            'product_title (use shopify_product_variants.product_title instead)',
            'variant_id (use shopify_order_line_items.variant_id instead)',
            'quantity (use shopify_order_line_items.quantity instead)',
            'order_date (use created_at instead)'
        ],
        'enhanced_quirks': [
            '⚠️ CRITICAL: DO NOT use \'order_date\' - use \'created_at\' instead',
            'Date filtering: WHERE created_at BETWEEN :date_start AND :date_end',
            'This table contains ORDER-LEVEL data only, not product/line-item details',
            'To get product details, you MUST JOIN with shopify_order_line_items and shopify_product_variants'
        ]
    },
    'public.shopify_order_line_items': {
        'purpose': 'Order line items - individual products in each order',
        'alias_suggestions': ['oli', 'line_items'],
        'enhanced_key_columns': [
            {
                'name': 'item_id',
                'purpose': 'Primary key - unique identifier for each line item',
                'type': 'text',
                'primary_key': True
            },
            {
                'name': 'order_id',
                'purpose': 'Foreign key → shopify_orders.order_id (links to parent order)',
                'type': 'text',
                'foreign_key': True,
                'references': 'public.shopify_orders.order_id'
            },
            {
                'name': 'variant_id',
                'purpose': 'Foreign key → shopify_product_variants.variant_id (links to product variant)',
                'type': 'text',
                'foreign_key': True,
                'references': 'public.shopify_product_variants.variant_id'
            },
            {
                'name': 'quantity',
                'purpose': 'Quantity of items in this line item',
                'type': 'integer'
            },
            {
                'name': 'discounted_unit_price_amount',
                'purpose': 'Unit price after discounts (use this for revenue calculations)',
                'type': 'numeric'
            }
        ],
        'columns_not_in_table': [
            '⚠️ CRITICAL: sku - DOES NOT EXIST HERE! Use shopify_product_variants.sku instead',
            '⚠️ CRITICAL: product_id - DOES NOT EXIST HERE! Use shopify_product_variants.product_id instead',
            '⚠️ CRITICAL: product_title - DOES NOT EXIST HERE! Use shopify_product_variants.product_title instead'
        ],
        'enhanced_quirks': [
            '⚠️ CRITICAL: This table does NOT have \'sku\' column - it\'s in shopify_product_variants',
            '⚠️ CRITICAL: This table does NOT have \'product_id\' column - it\'s in shopify_product_variants',
            '⚠️ CRITICAL: This table does NOT have \'product_title\' column - it\'s in shopify_product_variants',
            'To get sku: JOIN public.shopify_product_variants pv ON oli.variant_id = pv.variant_id, then use pv.sku',
            'To get product_id: JOIN public.shopify_product_variants pv ON oli.variant_id = pv.variant_id, then use pv.product_id',
            'Revenue calculation: SUM(oli.quantity * oli.discounted_unit_price_amount)',
            'Always use variant_id to link to product_variants table'
        ]
    },
    'public.shopify_product_variants': {
        'purpose': 'Product variants - contains product-level information including SKU, product_id, and product_title',
        'alias_suggestions': ['pv', 'variants'],
        'enhanced_key_columns': [
            {
                'name': 'variant_id',
                'purpose': 'Primary key - unique identifier for each product variant',
                'type': 'text',
                'primary_key': True
            },
            {
                'name': 'product_id',
                'purpose': 'Product identifier - groups variants of the same product - THIS IS WHERE product_id IS!',
                'type': 'text'
            },
            {
                'name': 'product_title',
                'purpose': 'Product name/title - THIS IS WHERE product_title IS!',
                'type': 'text'
            },
            {
                'name': 'sku',
                'purpose': 'Stock keeping unit identifier - THIS IS WHERE SKU IS LOCATED!',
                'type': 'text'
            }
        ],
        'enhanced_quirks': [
            '⚠️ IMPORTANT: sku, product_id, and product_title are in THIS table, NOT in shopify_order_line_items',
            'To access sku from orders: JOIN public.shopify_order_line_items oli ON o.order_id = oli.order_id JOIN public.shopify_product_variants pv ON oli.variant_id = pv.variant_id, then use pv.sku',
            'product_id is available here - NOT in shopify_order_line_items',
            'To get product-level data, JOIN this table using variant_id from shopify_order_line_items'
        ]
    },
    'public.dw_meta_ads_attribution': {
        'purpose': 'Meta (Facebook) Ads attribution data - hourly granularity',
        'alias_suggestions': ['meta', 'fb_ads'],
        'enhanced_quirks': [
            'Date filtering: WHERE date_start BETWEEN :date_start AND :date_end',
            'Has hourly granularity - aggregate by date if daily data needed',
            'Use SUM() and GROUP BY date_start to get daily aggregates'
        ]
    },
    'public.dw_google_ads_attribution': {
        'purpose': 'Google Ads attribution data - hourly granularity',
        'alias_suggestions': ['google', 'gads'],
        'enhanced_quirks': [
            '⚠️ Uses \'cost_amount\' instead of \'spend\' (different from Meta Ads)',
            'Date filtering: WHERE date_start BETWEEN :date_start AND :date_end',
            'Has hourly granularity - aggregate by date if daily data needed'
        ]
    }
}


def enhance_metadata():
    """Enhance table metadata with detailed descriptions for key tables."""
    config_path = Path(__file__).parent.parent / "config" / "table_metadata.yaml"
    
    logger.info(f"📖 Loading metadata from {config_path}")
    with open(config_path, 'r') as f:
        metadata = yaml.safe_load(f) or {}
    
    tables = metadata.get('tables', {})
    logger.info(f"📊 Found {len(tables)} tables in metadata")
    
    # Enhance key tables
    enhanced_count = 0
    for table_name, enhancements in KEY_TABLE_ENHANCEMENTS.items():
        if table_name in tables:
            logger.info(f"✨ Enhancing {table_name}")
            table_meta = tables[table_name]
            
            # Update purpose
            if 'purpose' in enhancements:
                table_meta['purpose'] = enhancements['purpose']
            
            # Add alias suggestions
            if 'alias_suggestions' in enhancements:
                table_meta['alias_suggestions'] = enhancements['alias_suggestions']
            
            # Enhance key columns
            if 'enhanced_key_columns' in enhancements:
                # Merge with existing key_columns, preserving generated ones
                existing_key_cols = {col['name']: col for col in table_meta.get('key_columns', [])}
                for enhanced_col in enhancements['enhanced_key_columns']:
                    col_name = enhanced_col['name']
                    if col_name in existing_key_cols:
                        # Update existing column with enhanced info
                        existing_key_cols[col_name].update(enhanced_col)
                    else:
                        # Add new key column
                        if 'key_columns' not in table_meta:
                            table_meta['key_columns'] = []
                        table_meta['key_columns'].append(enhanced_col)
                table_meta['key_columns'] = list(existing_key_cols.values())
            
            # Add columns_not_in_table
            if 'columns_not_in_table' in enhancements:
                table_meta['columns_not_in_table'] = enhancements['columns_not_in_table']
            
            # Enhance quirks
            if 'enhanced_quirks' in enhancements:
                existing_quirks = set(table_meta.get('quirks', []))
                for quirk in enhancements['enhanced_quirks']:
                    existing_quirks.add(quirk)
                table_meta['quirks'] = list(existing_quirks)
            
            enhanced_count += 1
    
    # Save enhanced metadata
    logger.info(f"💾 Saving enhanced metadata...")
    with open(config_path, 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False, allow_unicode=True, width=120)
    
    logger.info(f"✅ Enhanced {enhanced_count} key tables")
    logger.info(f"📊 Total tables: {len(tables)}")
    print(f"\n✅ Enhanced metadata for {enhanced_count} key tables")
    print(f"📊 Total tables in metadata: {len(tables)}")


if __name__ == "__main__":
    enhance_metadata()


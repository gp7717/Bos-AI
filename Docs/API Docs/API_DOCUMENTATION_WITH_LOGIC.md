# Complete API Documentation with Logic & Functionality

This document provides a comprehensive list of all APIs with detailed explanations of their logic, functionality, and implementation details.

## Base URL

- Production: `https://dashboard.seleric.ai/api`
- All routes under `/api` require authentication (except `/health` and `/metrics`)

---

## 1. Server Health & Metrics (No Authentication Required)

### GET `/health`

**Functionality**: Server health check endpoint that verifies database connectivity and server status.

**Logic**:

- Executes a simple database query (`SELECT 1`) to verify PostgreSQL connection
- Returns server status, database connection status, timestamp, and uptime
- Returns 503 status if database connection fails

**Response**:

```json
{
  "status": "healthy",
  "database": "connected",
  "timestamp": "2025-01-XX...",
  "uptime": 12345.67
}
```

---

### GET `/metrics`

**Functionality**: Returns server performance metrics.

**Logic**:

- Retrieves process uptime from Node.js runtime
- Returns memory usage statistics (heap used, heap total, external, rss)
- Provides timestamp for metrics snapshot

**Response**:

```json
{
  "uptime": 12345.67,
  "memory": { "heapUsed": ..., "heapTotal": ..., "external": ..., "rss": ... },
  "timestamp": "2025-01-XX..."
}
```

---

### GET `/`

**Functionality**: Simple root endpoint to verify server is running.

**Logic**:

- Returns plain text message "Server is running"
- No database queries or complex operations

---

## 2. Product Metrics (Direct Routes - Authentication Required)

### GET `/product_metrics`

**Functionality**: Retrieves all product variants with calculated margin.

**Logic**:

- Joins `variants_tt` and `products_tt` tables
- Calculates margin for each variant: `margin = Selling_Price - COGS`
- Returns variant details including SKU, price, unit cost, and calculated margin
- Handles null values gracefully

**Data Flow**:

1. Query database for all variants with product information
2. Calculate margin for each row
3. Format currency values

---

### POST `/product_metrics`

**Functionality**: Creates a new product variant (product metric).

**Logic**:

1. Accepts product data: `product_name`, `variant_title`, `sku_name`, `selling_price`, `cogs`
2. Looks up `product_id` from `products_tt` table using `product_name`
3. Inserts new record into `variants_tt` table with product_id reference
4. Returns created variant record

**Validation**:

- Validates product exists before creating variant
- Returns 400 if product not found

---

### PUT `/product_metrics/:sku_name`

**Functionality**: Updates an existing product variant by SKU.

**Logic**:

1. Finds product_id from product_name
2. Updates variant record matching the SKU
3. Updates product_id, variant_title, price, and unit_cost
4. Returns updated record or 404 if not found

---

### DELETE `/product_metrics/:sku_name`

**Functionality**: Deletes a product variant by SKU.

**Logic**:

- Deletes record from `variants_tt` where SKU matches
- Returns success message or 404 if variant not found

---

## 3. Core Analytics APIs

### GET `/api/ad_spend`

**Functionality**: Calculates total ad spend from Meta (Facebook) and Google Ads for a date range.

**Logic**:

1. Accepts optional `startDate` and `endDate` query parameters (defaults to today)
2. Calls `adSpendService.fetchAllAdSpend()` which:
   - **Google Ads**: Makes API call to Google Ads API with query for `metrics.cost_micros` (converted from micros to dollars)
   - **Meta Ads**: Makes API call to Facebook Graph API for account-level insights (spend field)
   - Executes both API calls in parallel using `Promise.all()`
3. Returns breakdown: `{ googleSpend, facebookSpend, totalSpend }`

**Data Sources**:

- Google Ads API (v20)
- Facebook Graph API (v19.0)

**Error Handling**: Returns 500 with error details if API calls fail

---

### GET `/api/cogs`

**Functionality**: Calculates total Cost of Goods Sold (COGS) from Shopify orders, categorized by traffic source.

**Logic**:

1. Accepts `startDate` and `endDate` (defaults to today)
2. Calls `fetchTotalCogs()` helper which:
   - Makes GraphQL query to Shopify Admin API for orders in date range
   - Filters out cancelled orders
   - Extracts UTM source from custom attributes
   - For each line item: calculates `unitCost * quantity`
   - Categorizes COGS by source:
     - **Meta**: facebook, instagram, meta, fb, ig, IGShopping
     - **Google**: utm_source = "google"
     - **Organic**: everything else (no UTM or other sources)
   - Handles pagination (50 orders per page)
3. Returns: `{ metaCogs, googleCogs, organicCogs, totalCogs }`

**Data Source**: Shopify GraphQL Admin API

---

### GET `/api/net_profit`

**Functionality**: Calculates net profit by subtracting COGS and ad spend from sales revenue.

**Logic**:

1. Fetches three data sources in parallel:
   - Total sales (by source: Meta, Google, Organic)
   - Total COGS (by source: Meta, Google, Organic)
   - Total ad spend (Meta + Google)
2. Calculates net profit for each channel:
   - `metaNetProfit = metaSales - metaCogs - facebookSpend`
   - `googleNetProfit = googleSales - googleCogs - googleSpend`
   - `totalNetProfit = totalSales - totalCogs - totalAdSpend`
   - `organicNetProfit = organicSales - organicCogs` (no ad spend)
3. Returns breakdown by channel

**Formula**: `Net Profit = Revenue - COGS - Ad Spend`

---

### GET `/api/sales`

**Functionality**: Retrieves total sales revenue from Shopify orders, categorized by traffic source.

**Logic**:

1. Calls `fetchTotalSales()` helper:
   - Makes GraphQL query to Shopify for orders in date range
   - Extracts UTM source from custom attributes
   - Sums order totals (excluding cancelled orders)
   - Categorizes by source (Meta, Google, Organic) using same logic as COGS
2. Returns: `{ metaSales, googleSales, organicSales, totalSales }`

**Data Source**: Shopify GraphQL Admin API

---

### GET `/api/roas`

**Functionality**: Calculates Return on Ad Spend (ROAS) metrics for Meta and Google.

**Logic**:

1. Fetches sales, COGS, and ad spend data
2. Calculates three types of ROAS for each channel:
   - **Gross ROAS**: `Revenue / Ad Spend`
   - **Net ROAS**: `(Revenue - COGS) / Ad Spend`
   - **Break-Even ROAS**: `(COGS + Ad Spend) / Ad Spend`
3. Uses safe division (returns null if ad spend is 0)
4. Returns breakdown for Meta, Google, and Total

**Formulas**:

- Gross ROAS = Total Revenue / Ad Spend
- Net ROAS = (Revenue - COGS) / Ad Spend
- BE ROAS = (COGS + Ad Spend) / Ad Spend

---

### GET `/api/roas_by_date`

**Functionality**: Calculates ROAS metrics broken down by date within a date range.

**Logic**:

- Fetches daily breakdown of sales, COGS, and ad spend
- Calculates ROAS metrics for each day
- Returns array of daily ROAS data with date, revenue, COGS, ad spend, and ROAS metrics

---

### GET `/api/net_profit_single_day`

**Functionality**: Calculates net profit for a single day or date range with detailed breakdown.

**Logic**:

1. Accepts either:
   - `date` parameter for single day
   - `startDate` and `endDate` for date range
2. For single day:
   - Queries database for revenue and COGS from `shopify_orders` table
   - Fetches ad spend from ad spend service
   - Calculates: `netProfit = revenue - cogs - adSpend`
3. For date range:
   - Calculates daily breakdowns
   - Aggregates totals across all days
4. Uses `product_variant_cost_history` table for accurate COGS based on date

**Database Queries**:

- Revenue: Sum of `total_price_amount` from `shopify_orders`
- COGS: Sum of `quantity * unit_cost_amount` from joined tables with cost history

---

### GET `/api/order_count`

**Functionality**: Counts total orders and quantities, categorized by traffic source.

**Logic**:

1. Makes GraphQL query to Shopify for orders
2. Filters out cancelled orders
3. Counts:
   - Total order count
   - Total quantity of items
   - Quantity by source (Meta, Google, Organic) based on UTM source
4. Handles pagination (50 orders per page)

**Returns**: `{ orderCount, totalQuantity, metaQuantity, googleQuantity, organicQuantity }`

---

### GET `/api/orders/:timeframe`

**Functionality**: Retrieves orders grouped by timeframe (daily, weekly, monthly).

**Logic**:

- Accepts timeframe parameter: `daily`, `weekly`, or `monthly`
- Queries orders from database
- Groups orders by the specified time period
- Returns order counts and potentially order details grouped by period

---

### GET `/api/latest_orders`

**Functionality**: Retrieves the most recent orders from Shopify.

**Logic**:

- Queries Shopify GraphQL API for recent orders
- Orders by creation date (reverse chronological)
- Returns limited set of latest orders with order details

---

### GET `/api/net_sales/:timeframe`

**Functionality**: Calculates net sales (revenue) grouped by timeframe.

**Logic**:

- Groups orders by timeframe (daily/weekly/monthly)
- Calculates net sales for each period
- Returns array of periods with net sales amounts

---

### GET `/api/products`

**Functionality**: Retrieves all products from Shopify.

**Logic**:

- Makes GraphQL query to Shopify Admin API
- Retrieves product information including variants, SKUs, prices
- Returns product list

---

### POST `/api/products/sync`

**Functionality**: Synchronizes products from Shopify to local database.

**Logic**:

- Fetches all products from Shopify
- Updates or inserts product data into local database
- Handles variants, SKUs, and product metadata
- Ensures data consistency between Shopify and local storage

---

## 4. Geographic Analytics APIs

### GET `/api/order_count_by_province`

**Functionality**: Counts orders grouped by province/state.

**Logic**:

- Queries orders from database
- Groups by `billing_province_name` or `shipping_province_name`
- Counts orders per province
- Returns province name and order count

---

### GET `/api/order_sales_by_province`

**Functionality**: Calculates total sales revenue grouped by province.

**Logic**:

- Groups orders by province
- Sums order totals for each province
- Returns province name and total sales amount

---

### GET `/api/top_skus_by_sales`

**Functionality**: Identifies top-selling SKUs by sales revenue.

**Logic**:

- Queries order line items from database
- Groups by SKU
- Calculates total sales (quantity \* price) for each SKU
- Orders by total sales descending
- Returns top N SKUs with sales data

---

## 5. Source Analytics APIs

### GET `/api/source_visitor/:timeframe`

**Functionality**: Analyzes visitor traffic by source for a given timeframe.

**Logic**:

- Accepts timeframe parameter (daily/weekly/monthly)
- Queries order data or analytics data
- Groups visitors/traffic by UTM source
- Returns visitor counts by source (Meta, Google, Organic, etc.)

---

### GET `/api/ad_spend_by_hour`

**Functionality**: Breaks down ad spend by hour of day.

**Logic**:

1. Calls `adSpendService.fetchFacebookAdSpendHourly()` for hourly Facebook data
2. Aggregates ad spend by hour (0-23)
3. Returns hourly breakdown showing spend patterns throughout the day

**Data Source**: Facebook Graph API with hourly breakdown

---

### GET `/api/product_spend_and_sales_by_title`

**Functionality**: Matches ad spend to product sales by product title.

**Logic**:

1. Fetches product-level ad spend from Meta/Google (by product/campaign)
2. Fetches product sales from Shopify orders
3. Matches products by title/name
4. Calculates:
   - Ad spend per product
   - Sales revenue per product
   - ROAS per product
5. Returns array of products with spend and sales data

**Use Case**: Identify which products are most profitable from ad spend perspective

---

### GET `/api/meta_entity_report`

**Functionality**: Generates comprehensive Meta (Facebook) entity report with attribution data.

**Logic**:

1. Accepts `startDate` and `endDate` query parameters
2. Calls `EntityReportService` which:
   - Fetches Meta ads data (campaigns, ad sets, ads, creatives)
   - Fetches Shopify orders with UTM parameters
   - Matches orders to ads using UTM campaign, content, and source
   - Groups data by entity hierarchy: Campaign → Ad Set → Ad → Creative
   - Calculates metrics: revenue, COGS, ad spend, net profit, ROAS
   - Returns detailed breakdown with hourly data

**Data Structure**:

- Campaigns → Ad Sets → Ads → Creatives
- Each level contains spend, sales, profit metrics
- Hourly breakdown for attribution analysis

---

### GET `/api/meta_entity_report_hierarchy`

**Functionality**: Returns Meta entity report in hierarchical structure optimized for tree visualization.

**Logic**:

- Similar to entity report but structures data as nested tree
- Optimized for frontend tree/graph components
- Includes parent-child relationships between entities

---

### GET `/api/organic_entity_report`

**Functionality**: Generates report for organic (non-paid) traffic attribution.

**Logic**:

1. Calls `OrganicReportService`:
   - Fetches orders without UTM source or with organic sources
   - Groups by time periods (hourly)
   - Calculates organic revenue and COGS
   - Matches orders to time windows for attribution
2. Returns summary and hourly breakdown of organic sales

**Attribution Model**: Time-based attribution (orders matched to time windows)

---

### GET `/api/google_entity_report`

**Functionality**: Generates comprehensive Google Ads entity report with detailed attribution.

**Logic**:

1. Calls `GoogleEntityReportService`:
   - Fetches Google Ads data: Campaigns → Ad Groups → Ads → Keywords
   - Fetches Shopify orders with Google UTM parameters
   - Matches orders to ads using UTM campaign, content, term
   - Uses Google Ads conversion tracking data
   - Groups by entity hierarchy
   - Calculates attribution using Google's conversion data and UTM matching
2. Returns detailed breakdown with campaign, ad group, ad, and keyword level data

**Attribution**: Combines Google conversion tracking with UTM parameter matching

---

### GET `/api/google_entity_report_aggregated`

**Functionality**: Returns aggregated Google Ads report with summarized metrics.

**Logic**:

- Similar to entity report but aggregates data at higher levels
- Summarizes metrics across campaigns, ad groups
- Provides roll-up totals for easier analysis
- Optimized for high-level performance overview

---

### GET `/api/amazon_daily_metrics`

**Functionality**: Retrieves daily metrics from Amazon Ads.

**Logic**:

1. Fetches Amazon Ads data from Amazon Ads API
2. Retrieves daily spend, impressions, clicks, sales data
3. Calculates metrics: ROAS, ACOS (Advertising Cost of Sale), etc.
4. Returns daily breakdown of Amazon advertising performance

**Data Source**: Amazon Ads API

---

## 6. Amazon Entity Report APIs

### GET `/api/amazon_entity_report`

**Functionality**: Generates comprehensive Amazon Ads entity report.

**Logic**:

1. Accepts `startDate` and `endDate` query parameters
2. Calls `AmazonEntityReportService`:
   - Fetches Amazon Ads report data (campaigns, ad groups, keywords, products)
   - Fetches order/sales data from Amazon
   - Matches ads to sales for attribution
   - Groups by entity hierarchy: Campaign → Ad Group → Keyword/Product
   - Calculates metrics: spend, sales, profit, ROAS
3. Returns detailed entity report with attribution data

**Data Source**: Amazon Ads API reports

---

## 7. Content Generation APIs

### GET `/api/content-generation/test`

**Functionality**: Simple test endpoint to verify content generation routes are working.

**Logic**:

- Returns JSON message confirming routes are functional
- No external API calls

---

### GET `/api/content-generation/content/generated`

**Functionality**: Proxies request to Python backend for generated content.

**Logic**:

1. Makes HTTP GET request to Python API (`PYTHON_API_URL`)
2. Proxies response back to client
3. Handles errors from Python backend gracefully

**Architecture**: Acts as proxy between Node.js backend and Python content generation service

---

## 8. Customer Order APIs

### POST `/api/customer-orders`

**Functionality**: Creates a new customer order record in the database.

**Logic**:

1. Validates `order_id` is provided
2. Calls `customerDetailsService.insertCustomerOrder()`:
   - Transforms customer data to database format using `CustomerDataTransformer`
   - Validates required fields and data types
   - Sanitizes input data
   - Checks BlueDart service availability for shipping pincode
   - Inserts into `customer_details_th` table
   - Uses `ON CONFLICT` to update if order_id already exists (upsert)
3. Returns success with order_id or validation errors

**Data Storage**: PostgreSQL table `customer_details_th` with 42+ fields including billing, shipping, payment, and line items

**Validation**: Validates email, phone, addresses, payment data, and line items

---

### POST `/api/customer-orders/bulk`

**Functionality**: Creates multiple customer orders in a single request.

**Logic**:

1. Accepts array of customer order objects
2. Processes each order:
   - Validates and transforms data
   - Checks for duplicates
   - Inserts/updates in database
3. Returns results with success/failure counts and details

**Performance**: Processes in batches to optimize database performance

---

### GET `/api/customer-orders`

**Functionality**: Retrieves customer orders with filtering and pagination.

**Logic**:

1. Accepts query parameters:
   - `page`, `limit` for pagination
   - Filters: date range, status, province, etc.
2. Queries `customer_details_th` table with filters
3. Returns paginated results with order details

**Filtering**: Supports filtering by date, status, province, payment method, etc.

---

### GET `/api/customer-orders/search`

**Functionality**: Flexible search across customer orders with multiple search criteria.

**Logic**:

- Accepts search parameters: order_id, email, phone, name, address, etc.
- Builds dynamic SQL query based on provided search terms
- Supports partial matching and multiple criteria
- Returns matching orders

---

### GET `/api/customer-orders/:order_id`

**Functionality**: Retrieves a single customer order by order ID.

**Logic**:

- Queries database for order matching `order_id`
- Returns full order details including line items, billing, shipping, payment info
- Returns 404 if order not found

---

### PUT `/api/customer-orders/:order_id`

**Functionality**: Updates an existing customer order.

**Logic**:

1. Validates order exists
2. Transforms and validates updated data
3. Updates record in database (partial updates supported)
4. Returns updated order data

---

### DELETE `/api/customer-orders/:order_id`

**Functionality**: Deletes a customer order from database.

**Logic**:

- Deletes record from `customer_details_th` table
- Returns success message or 404 if not found

---

### GET `/api/customer-orders/stats`

**Functionality**: Calculates statistics for customer orders.

**Logic**:

- Aggregates order data:
  - Total orders count
  - Total revenue
  - Average order value
  - Orders by status
  - Orders by province
  - Revenue by time period
- Returns summary statistics

---

## 9. Shipping - Waybill APIs

### POST `/api/shipping/generate-waybill`

**Functionality**: Generates a single waybill using BlueDart shipping API.

**Logic**:

1. Validates waybill data structure
2. Extracts `CreditReferenceNo` (order_name) from services
3. Calls `shippingService.generateWaybill()`:
   - Checks BlueDart service availability for pincode and payment method
   - Retrieves customer data to check outstanding balance
   - Gets JWT token from BlueDart API
   - Transforms data to BlueDart API format
   - Makes API call to BlueDart to generate waybill
   - Stores waybill data in database (AWB number, PDF, etc.)
   - Handles duplicate waybill prevention
4. Returns waybill details including AWB number and PDF content

**Integration**: BlueDart API for waybill generation
**Data Storage**: Stores waybill in database for tracking and PDF retrieval

---

### POST `/api/shipping/generate-bulk-waybills`

**Functionality**: Generates multiple waybills in a single request.

**Logic**:

1. Accepts array of waybill data objects
2. Processes each waybill:
   - Validates data
   - Checks service availability
   - Generates waybill via BlueDart API
   - Stores in database
3. Returns results with success/failure counts and details for each

**Error Handling**: Continues processing even if individual waybills fail

---

### POST `/api/shipping/import-data`

**Functionality**: Imports bulk data for waybill generation operations.

**Logic**:

- Accepts bulk data in specified format
- Validates and transforms data
- Prepares data for bulk waybill generation
- Returns import results

---

### GET `/api/shipping/waybill/:trackingNumber`

**Functionality**: Retrieves waybill details from database by tracking number.

**Logic**:

- Queries database for waybill matching tracking number (AWB)
- Returns complete waybill data including:
  - Consignee and shipper details
  - Services and payment method
  - AWB number
  - PDF content
  - Status information

---

### GET `/api/shipping/waybills`

**Functionality**: Retrieves all waybills with pagination and filtering.

**Logic**:

1. Accepts query parameters:
   - `page`, `limit` for pagination
   - Filters: date range, status, destination, etc.
2. Queries waybill database with filters
3. Returns paginated list of waybills

---

### GET `/api/shipping/track/:trackingNumber`

**Functionality**: Tracks shipment status using BlueDart tracking API.

**Logic**:

1. Calls BlueDart tracking API with AWB number
2. Retrieves current shipment status and tracking events
3. Returns tracking information including:
   - Current status
   - Location
   - Estimated delivery
   - Tracking history

**Integration**: BlueDart tracking API

---

### POST `/api/shipping/cancel-waybill`

**Functionality**: Cancels an existing waybill.

**Logic**:

1. Validates waybill exists and is cancellable
2. Calls BlueDart API to cancel waybill
3. Updates waybill status in database
4. Returns cancellation confirmation

---

### POST `/api/shipping/update-ewaybill`

**Functionality**: Updates e-waybill information for GST compliance.

**Logic**:

1. Accepts e-waybill data (invoice details, GST information)
2. Updates e-waybill via BlueDart API or directly in system
3. Stores updated e-waybill data
4. Returns update confirmation

**Purpose**: Maintains GST compliance for shipments

---

### GET `/api/shipping/stats`

**Functionality**: Calculates statistics for waybills.

**Logic**:

- Aggregates waybill data:
  - Total waybills generated
  - Waybills by status
  - Waybills by destination
  - Average delivery time
  - Revenue from shipping
- Returns summary statistics

---

## 10. Shipping - Download APIs

### GET `/api/shipping/download-pdf/:trackingNumber`

**Functionality**: Downloads waybill PDF for a specific tracking number.

**Logic**:

1. Retrieves waybill from database
2. Extracts PDF content (stored as base64 or blob)
3. Returns PDF file with appropriate headers
4. Handles PDF generation if not stored

---

### GET `/api/shipping/download-th-pdf/:trackingNumber`

**Functionality**: Downloads Thailand-specific waybill PDF format.

**Logic**:

- Similar to regular PDF download but uses Thailand-specific PDF template
- Formats data according to Thailand shipping requirements

---

### POST `/api/shipping/download-bulk-pdf`

**Functionality**: Downloads multiple waybill PDFs as a ZIP file.

**Logic**:

1. Accepts array of tracking numbers
2. Retrieves PDFs for each waybill
3. Creates ZIP file containing all PDFs
4. Returns ZIP file for download

---

### POST `/api/shipping/download-bulk-th-pdf`

**Functionality**: Downloads multiple Thailand waybill PDFs as a merged PDF.

**Logic**:

1. Accepts array of tracking numbers
2. Retrieves Thailand PDFs for each waybill
3. Merges PDFs into single document
4. Returns merged PDF file

---

## 11. Shipping - Pincode APIs

### GET `/api/shipping/check-pincode/:pincode`

**Functionality**: Checks if BlueDart service is available for a specific pincode.

**Logic**:

1. Calls BlueDart API to check service availability
2. Checks different payment methods (prepaid, COD, etc.)
3. Returns availability status and service details
4. May cache results in database

**Purpose**: Validate shipping availability before order processing

---

### POST `/api/shipping/check-bulk-pincodes`

**Functionality**: Checks service availability for multiple pincodes.

**Logic**:

1. Accepts array of pincodes
2. Checks availability for each pincode
3. Returns results for all pincodes
4. Optimizes API calls through batching

---

## 12. Database Analytics APIs

### GET `/api/net_profit_daily`

**Functionality**: Calculates net profit for the last N days with daily breakdown.

**Logic**:

1. Accepts `n` query parameter (defaults to 7)
2. Calls `getLastNDaysNetProfit()` helper:
   - Queries `shopify_orders` table for last N days
   - Calculates daily revenue, COGS, ad spend
   - Calculates daily net profit
3. Returns array of daily net profit data

**Data Source**: Local PostgreSQL database (`shopify_orders` table)

---

### GET `/api/sales_unitCost_by_hour`

**Functionality**: Breaks down sales and unit cost by hour of day.

**Logic**:

1. Queries orders grouped by hour (0-23)
2. Calculates:
   - Total sales per hour
   - Total unit cost per hour
   - Average order value per hour
3. Returns hourly breakdown

**Use Case**: Identify peak sales hours and optimize ad spend timing

---

### POST `/api/historical_stats_by_date`

**Functionality**: Retrieves historical statistics for a date range.

**Logic**:

1. Accepts `startDate` and `endDate` in request body
2. Queries database for orders in date range
3. Calculates:
   - Total revenue
   - Total COGS
   - Order count
   - Total quantity
   - Cancelled orders stats
4. Returns comprehensive historical statistics

**Data Source**: `shopify_orders` and related tables

---

## 13. User Management APIs

### POST `/api/users`

**Functionality**: Creates a new user in Firebase Auth and Firestore.

**Logic**:

1. Validates required fields: email, displayName, role
2. Checks role assignment permissions (super_admin can create any role, admin can create manager/user, etc.)
3. Checks if user with email already exists
4. Creates user in Firebase Auth with temporary password
5. Generates email verification link
6. Creates user document in Firestore with:
   - Email, displayName, role
   - Status: "pending_verification"
   - Timestamps
7. Returns user details with verification link

**Roles**: super_admin, admin, manager, user
**Storage**: Firebase Auth + Firestore

---

### GET `/api/users`

**Functionality**: Retrieves all users from Firestore.

**Logic**:

- Queries Firestore `users` collection
- Returns user list with: uid, email, displayName, role, status, timestamps
- Requires admin or super_admin role

---

### GET `/api/users/uid/:uid`

**Functionality**: Retrieves a specific user by UID.

**Logic**:

- Queries Firestore for user document with matching UID
- Returns user details including sidebar permissions
- Returns 404 if user not found

---

### PUT `/api/users/:userId/role`

**Functionality**: Updates user role and/or sidebar permissions.

**Logic**:

1. Validates role (if provided) - only super_admin can change roles
2. Validates sidebar permissions structure
3. Normalizes sidebar permissions to new format:
   - Old format: boolean
   - New format: `{ enabled: boolean, operations: [] }`
4. Checks role assignment permissions
5. Prevents self-promotion to higher roles
6. Updates Firestore user document
7. Returns updated user data

**Sidebar Permissions**: Controls access to dashboard sections (dashboard, skuList, procurement, etc.)

---

### DELETE `/api/users/:userId`

**Functionality**: Deletes a user from Firebase Auth and Firestore.

**Logic**:

1. Prevents users from deleting themselves
2. Deletes user from Firebase Auth
3. Deletes user document from Firestore
4. Returns success confirmation

---

### GET `/api/user/role`

**Functionality**: Retrieves current authenticated user's role and permissions.

**Logic**:

1. Extracts user UID from Firebase Auth token
2. Queries Firestore for user document
3. Returns:
   - User role
   - Sidebar permissions (or null for defaults)
   - User details (uid, email, emailVerified)

**Authentication**: Uses Firebase Auth middleware

---

### POST `/api/users/verify-email`

**Functionality**: Verifies user existence by email address.

**Logic**:

1. Searches Firestore for user with matching email
2. Returns user details including UID
3. Returns 404 if user not found

**Use Case**: Lookup user UID by email for admin operations

---

## 14. Procurement APIs

### POST `/api/procurement/products`

**Functionality**: Creates a new product with variants, images, and vendors.

**Logic**:

1. Accepts product data, variants array, image requests, vendors array
2. Processes image requests:
   - Ensures only one primary image per product
   - Sets sort order for images
3. Calls `procurementProductService.createProductWithDetails()`:
   - Creates product record
   - Creates variant records linked to product
   - Creates vendor records linked to product
   - Creates image placeholder records
4. Generates Azure Blob Storage SAS URLs for image uploads
5. Returns product details with upload URLs

**Image Upload Flow**: Two-step process - generate upload URLs, then confirm after upload

---

### GET `/api/procurement/products`

**Functionality**: Retrieves all products with pagination.

**Logic**:

- Accepts `page` and `limit` query parameters (max 100 per page)
- Queries database for products with variants, images, vendors
- Returns paginated product list with metadata

---

### GET `/api/procurement/products/search`

**Functionality**: Searches products by various criteria.

**Logic**:

- Accepts search parameters: name, SKU, vendor, status, etc.
- Builds dynamic query based on search terms
- Returns matching products

---

### GET `/api/procurement/products/:productId`

**Functionality**: Retrieves a single product by ID with full details.

**Logic**:

- Queries database for product with all related data:
  - Product details
  - Variants with pricing
  - Images with URLs
  - Vendors with pricing
- Returns complete product object

---

### PUT `/api/procurement/products/:productId`

**Functionality**: Updates an existing product.

**Logic**:

1. Validates product exists
2. Updates product fields
3. Handles variant updates (create/update/delete)
4. Handles vendor updates
5. Returns updated product

---

### DELETE `/api/procurement/products/:productId`

**Functionality**: Deletes a product and all related data.

**Logic**:

- Deletes product record
- Cascades to variants, images, vendors
- May soft delete (mark as deleted) instead of hard delete

---

### PUT `/api/procurement/products/:productId/status`

**Functionality**: Updates product status (active, inactive, etc.).

**Logic**:

- Updates status field in product record
- Returns updated product

---

### DELETE `/api/procurement/variants/:variantId`

**Functionality**: Deletes a product variant.

**Logic**:

- Deletes variant record
- Handles related data cleanup (pricing, inventory references)

---

### POST `/api/procurement/products/:productId/images/upload-urls`

**Functionality**: Generates secure Azure Blob Storage upload URLs for images.

**Logic**:

1. Validates image requests (count, file types)
2. Generates SAS (Shared Access Signature) URLs for Azure Blob Storage
3. Creates image records in database with "pending" status
4. Returns upload URLs with expiration times

**Security**: Time-limited SAS URLs with write permissions only

---

### POST `/api/procurement/images/:imageId/confirm`

**Functionality**: Confirms image upload and updates image record.

**Logic**:

1. Validates image was successfully uploaded to Azure
2. Updates image record status to "confirmed"
3. Updates image URL in database
4. Handles primary image assignment

---

### GET `/api/procurement/images/:imageId/view`

**Functionality**: Generates secure view URL for image.

**Logic**:

- Generates time-limited SAS URL for reading image from Azure
- Returns URL for image display

---

### DELETE `/api/procurement/images/:imageId`

**Functionality**: Deletes image from Azure and database.

**Logic**:

1. Deletes image from Azure Blob Storage
2. Deletes image record from database
3. Handles cleanup of image references

---

### DELETE `/api/procurement/images/cleanup-pending`

**Functionality**: Cleans up images stuck in "pending" status.

**Logic**:

- Finds images with "pending" status older than threshold
- Deletes from Azure and database
- Returns cleanup results

---

### DELETE `/api/procurement/images/cleanup-orphaned`

**Functionality**: Cleans up orphaned images in Azure not referenced in database.

**Logic**:

1. Lists all images in Azure Blob Storage
2. Checks which are not in database
3. Deletes orphaned images
4. Returns cleanup statistics

---

### GET `/api/procurement/storage/stats`

**Functionality**: Retrieves Azure Blob Storage statistics.

**Logic**:

- Calculates:
  - Total storage used
  - Number of images
  - Storage by container/folder
  - Cost estimates
- Returns storage metrics

---

### POST `/api/procurement/products/:productId/vendors`

**Functionality**: Creates a vendor record linked to a product.

**Logic**:

1. Accepts vendor data: name, pricing, MOQ (Minimum Order Quantity), etc.
2. Creates vendor record linked to product
3. Stores vendor-specific pricing for product variants
4. Returns vendor details

---

### GET `/api/procurement/products/:productId/vendors`

**Functionality**: Retrieves all vendors for a product.

**Logic**:

- Queries database for vendors linked to product
- Returns vendor list with pricing information

---

### GET `/api/procurement/vendors/:vendorId`

**Functionality**: Retrieves a vendor by ID.

**Logic**:

- Queries database for vendor with all details
- Returns vendor information including product associations

---

### PUT `/api/procurement/vendors/:vendorId`

**Functionality**: Updates vendor information.

**Logic**:

- Updates vendor fields (name, contact, pricing, etc.)
- Returns updated vendor

---

### DELETE `/api/procurement/vendors/:vendorId`

**Functionality**: Deletes a vendor record.

**Logic**:

- Deletes vendor record
- Handles cleanup of product associations

---

### GET `/api/procurement/vendors/search`

**Functionality**: Searches vendors by various criteria.

**Logic**:

- Accepts search parameters
- Returns matching vendors

---

## 15. Master Vendor APIs

### POST `/api/masters/vendor/`

**Functionality**: Creates a new vendor in the master vendor list.

**Logic**:

1. Validates required fields: vendor_name, vendor_address, vendor_phone_no, vendor_gst_number
2. Checks for duplicate vendor name
3. Checks for duplicate GST number
4. Creates vendor record in master vendor table
5. Returns created vendor

**Purpose**: Central vendor master data management (separate from product-specific vendors)

---

### GET `/api/masters/vendor/`

**Functionality**: Retrieves all vendors with pagination.

**Logic**:

- Accepts `page`, `limit`, and optional `status` filter
- Queries master vendor table
- Returns paginated vendor list

---

### GET `/api/masters/vendor/:vendorId`

**Functionality**: Retrieves a vendor by ID.

**Logic**:

- Queries database for vendor
- Returns vendor details

---

### PUT `/api/masters/vendor/:vendorId`

**Functionality**: Updates vendor information.

**Logic**:

- Validates vendor exists
- Updates vendor fields
- Returns updated vendor

---

### DELETE `/api/masters/vendor/:vendorId`

**Functionality**: Deletes a vendor from master list.

**Logic**:

- Deletes vendor record
- May check for existing references before deletion

---

## 16. Master Product APIs

### POST `/api/masters/product/`

**Functionality**: Creates a new product in the master product list.

**Logic**:

1. Validates required product fields
2. Checks for duplicate product names/SKUs
3. Creates product record in master product table
4. Returns created product

**Purpose**: Central product master data management

---

### GET `/api/masters/product/`

**Functionality**: Retrieves all products with pagination.

**Logic**:

- Accepts pagination parameters
- Queries master product table
- Returns paginated product list

---

### GET `/api/masters/product/:productId`

**Functionality**: Retrieves a product by ID.

**Logic**:

- Queries database for product
- Returns product details

---

### PUT `/api/masters/product/:productId`

**Functionality**: Updates product information.

**Logic**:

- Validates product exists
- Updates product fields
- Returns updated product

---

### DELETE `/api/masters/product/:productId`

**Functionality**: Deletes a product from master list.

**Logic**:

- Deletes product record
- May check for existing references

---

## 17. Receiving - Purchase Request APIs

### POST `/api/receiving/purchase-request/`

**Functionality**: Creates a new purchase request.

**Logic**:

1. Accepts purchase request data:
   - Product/variant information
   - Quantity requested
   - Vendor information
   - Requested delivery date
   - Notes
2. Creates purchase request record with status "pending"
3. Returns created request

**Status Flow**: pending → approved → received → completed

---

### GET `/api/receiving/purchase-request/`

**Functionality**: Retrieves all purchase requests with pagination.

**Logic**:

- Accepts pagination and filter parameters
- Queries purchase request table
- Returns paginated list with request details

---

### GET `/api/receiving/purchase-request/:requestId`

**Functionality**: Retrieves a purchase request by ID.

**Logic**:

- Queries database for request with all details
- Returns request including associated quality checks

---

### PUT `/api/receiving/purchase-request/:requestId`

**Functionality**: Updates a purchase request.

**Logic**:

- Updates request fields
- Returns updated request

---

### DELETE `/api/receiving/purchase-request/:requestId`

**Functionality**: Deletes a purchase request.

**Logic**:

- Deletes request record
- May cascade to related quality checks

---

### PATCH `/api/receiving/purchase-request/:requestId/status`

**Functionality**: Updates purchase request status.

**Logic**:

1. Validates status transition (e.g., can't go from completed to pending)
2. Updates status field
3. May trigger notifications or workflows
4. Returns updated request

---

## 18. Receiving - Quality Check APIs

### POST `/api/receiving/quality-check/bulk`

**Functionality**: Creates or updates multiple quality checks for a purchase request.

**Logic**:

1. Accepts array of quality check data:
   - Variant/SKU information
   - Quantity received
   - Quality status (passed/failed)
   - Notes
   - Defect information
2. Creates or updates quality check records
3. Links to purchase request
4. Returns results for all quality checks

---

### GET `/api/receiving/quality-check/request/:requestId`

**Functionality**: Retrieves all quality checks for a purchase request.

**Logic**:

- Queries database for quality checks linked to request
- Returns quality check list with details

---

### GET `/api/receiving/quality-check/:qualityCheckId`

**Functionality**: Retrieves a quality check by ID.

**Logic**:

- Queries database for quality check
- Returns check details including associated documents

---

### DELETE `/api/receiving/quality-check/:qualityCheckId`

**Functionality**: Deletes a quality check.

**Logic**:

- Deletes quality check record
- May cleanup associated documents

---

### POST `/api/receiving/quality-check/:requestId/documents`

**Functionality**: Uploads a document for a quality check.

**Logic**:

1. Accepts file upload (image, PDF, etc.)
2. Stores document in Azure Blob Storage or database
3. Creates document record linked to quality check
4. Returns document details with view URL

---

### GET `/api/receiving/quality-check/:requestId/documents`

**Functionality**: Lists all documents for a quality check.

**Logic**:

- Queries database for documents linked to request
- Returns document list with metadata

---

### DELETE `/api/receiving/quality-check/documents/:documentId`

**Functionality**: Deletes a quality check document.

**Logic**:

- Deletes document from storage
- Deletes document record from database

---

### GET `/api/receiving/quality-check/documents/:documentId/content`

**Functionality**: Streams document content for viewing/downloading.

**Logic**:

- Retrieves document from storage
- Streams content with appropriate content-type headers
- Supports range requests for large files

---

## 19. Stock Management APIs

### GET `/api/stock-management/variants/:variantId/inventory`

**Functionality**: Retrieves inventory information for a specific variant.

**Logic**:

1. Queries inventory data for variant:
   - Current stock quantity
   - Reserved quantity
   - Available quantity
   - Warehouse locations
   - Cost information
2. Calculates inventory metrics
3. Returns inventory details

**Data Sources**: Inventory tracking tables, stock movement history

---

### GET `/api/stock-management/variants/inventory`

**Functionality**: Retrieves inventory for all variants with pagination.

**Logic**:

- Accepts pagination parameters
- Queries inventory for all variants
- Returns paginated inventory list
- Includes summary statistics

---

### GET `/api/stock-management/returns/pending`

**Functionality**: Retrieves pending returns awaiting approval.

**Logic**:

1. Queries stock movement events with return status "pending"
2. Filters by approval status
3. Returns pending returns with:
   - Return details
   - Variant information
   - Quantity
   - Reason
   - Request date
4. Supports pagination

---

### GET `/api/stock-management/returns`

**Functionality**: Retrieves all returns with optional status filter.

**Logic**:

- Accepts optional `status` query parameter (pending, approved, rejected)
- Queries return events from stock movement table
- Returns paginated list of returns
- Includes all return statuses if no filter

---

### POST `/api/stock-management/returns/:eventId/approve`

**Functionality**: Approves a return request.

**Logic**:

1. Validates return exists and is in "pending" status
2. Updates return status to "approved"
3. Updates inventory:
   - Adds returned quantity to available stock
   - Records stock movement event
4. May trigger notifications
5. Returns approved return details

**Inventory Impact**: Increases available stock when return is approved

---

### POST `/api/stock-management/returns/:eventId/reject`

**Functionality**: Rejects a return request.

**Logic**:

1. Validates return exists and is in "pending" status
2. Updates return status to "rejected"
3. Records rejection reason
4. Does NOT update inventory
5. May trigger notifications
6. Returns rejected return details

---

## Summary

### Total APIs: 113

### Key Architectural Patterns:

1. **Data Aggregation**: Multiple APIs aggregate data from Shopify, Google Ads, Meta Ads, and Amazon Ads
2. **Attribution Logic**: Entity reports match orders to ads using UTM parameters and conversion tracking
3. **Service Availability**: Shipping APIs check BlueDart service availability before generating waybills
4. **Image Upload Flow**: Two-step process (generate URLs → confirm upload) for secure Azure Blob Storage uploads
5. **Role-Based Access**: Firebase Auth with Firestore for user management and permissions
6. **Upsert Pattern**: Customer orders use ON CONFLICT for insert or update
7. **Pagination**: Most list endpoints support pagination for performance
8. **Validation**: Comprehensive input validation and sanitization across all endpoints
9. **Error Handling**: Graceful error handling with detailed error messages
10. **Rate Limiting**: Different rate limits for different endpoint types

---

_Last Updated: Generated with detailed logic explanations_

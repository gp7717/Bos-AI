# Complete API List - Dashboard Backend

This document lists **ALL** APIs present in the backend, organized by category.

## Base URL

- Production: `https://dashboard.seleric.ai/api`
- All routes under `/api` require authentication (except `/health` and `/metrics`)

---

## 1. Server Health & Metrics (No Authentication Required)

### Health Check

- **GET** `/health` - Server health check with database connection status

### Metrics

- **GET** `/metrics` - Server metrics (uptime, memory usage)

### Root

- **GET** `/` - Server status message

---

## 2. Product Metrics (Direct Routes - Authentication Required)

- **GET** `/product_metrics` - Get all product metrics (variants with margin calculation)
- **POST** `/product_metrics` - Create a new product variant
- **PUT** `/product_metrics/:sku_name` - Update product variant by SKU
- **DELETE** `/product_metrics/:sku_name` - Delete product variant by SKU

---

## 3. Core Analytics APIs

### Financial Metrics

- **GET** `/api/ad_spend` - Get total ad spend
- **GET** `/api/cogs` - Get total cost of goods sold
- **GET** `/api/net_profit` - Get net profit
- **GET** `/api/sales` - Get total sales
- **GET** `/api/roas` - Get return on ad spend
- **GET** `/api/roas_by_date` - Get ROAS by date
- **GET** `/api/net_profit_single_day` - Get single day net profit

### Order Metrics

- **GET** `/api/order_count` - Get total order count
- **GET** `/api/orders/:timeframe` - Get orders by timeframe (daily/weekly/monthly)
- **GET** `/api/latest_orders` - Get latest orders
- **GET** `/api/net_sales/:timeframe` - Get net sales by timeframe

### Product Management

- **GET** `/api/products` - Get all products
- **POST** `/api/products/sync` - Sync Shopify products

---

## 4. Geographic Analytics APIs

- **GET** `/api/order_count_by_province` - Get order count by province
- **GET** `/api/order_sales_by_province` - Get order sales by province
- **GET** `/api/top_skus_by_sales` - Get top SKUs by sales

---

## 5. Source Analytics APIs

### Traffic & Visitors

- **GET** `/api/source_visitor/:timeframe` - Get source visitors by timeframe

### Ad Spend Analytics

- **GET** `/api/ad_spend_by_hour` - Get ad spend by hour
- **GET** `/api/product_spend_and_sales_by_title` - Get product spend and sales by title

### Entity Reports

- **GET** `/api/meta_entity_report` - Get Meta entity report
- **GET** `/api/meta_entity_report_hierarchy` - Get Meta entity report hierarchy
- **GET** `/api/organic_entity_report` - Get organic entity report
- **GET** `/api/google_entity_report` - Get Google entity report
- **GET** `/api/google_entity_report_aggregated` - Get aggregated Google entity report

### Amazon Analytics

- **GET** `/api/amazon_daily_metrics` - Get Amazon daily metrics

---

## 6. Amazon Entity Report APIs

- **GET** `/api/amazon_entity_report` - Get Amazon entity report (with date range query params: startDate, endDate)

---

## 7. Content Generation APIs

- **GET** `/api/content-generation/test` - Test content generation routes
- **GET** `/api/content-generation/content/generated` - Get generated content (proxies to Python backend)

---

## 8. Customer Order APIs

### Order Management

- **POST** `/api/customer-orders` - Create a new customer order
- **POST** `/api/customer-orders/bulk` - Bulk create customer orders
- **GET** `/api/customer-orders` - Get all customer orders (with filters)
- **GET** `/api/customer-orders/search` - Search customer orders (flexible search)
- **GET** `/api/customer-orders/:order_id` - Get customer order by ID
- **PUT** `/api/customer-orders/:order_id` - Update customer order
- **DELETE** `/api/customer-orders/:order_id` - Delete customer order

### Order Statistics

- **GET** `/api/customer-orders/stats` - Get customer order statistics

---

## 9. Shipping - Waybill APIs

### Waybill Generation

- **POST** `/api/shipping/generate-waybill` - Generate single waybill
- **POST** `/api/shipping/generate-bulk-waybills` - Generate bulk waybills
- **POST** `/api/shipping/import-data` - Import data for bulk operations

### Waybill Management

- **GET** `/api/shipping/waybill/:trackingNumber` - Get waybill by tracking number
- **GET** `/api/shipping/waybills` - Get all waybills (with pagination and filters)
- **GET** `/api/shipping/track/:trackingNumber` - Track shipment
- **POST** `/api/shipping/cancel-waybill` - Cancel waybill
- **POST** `/api/shipping/update-ewaybill` - Update e-waybill

### Waybill Statistics

- **GET** `/api/shipping/stats` - Get waybill statistics

---

## 10. Shipping - Download APIs

- **GET** `/api/shipping/download-pdf/:trackingNumber` - Download waybill PDF
- **GET** `/api/shipping/download-th-pdf/:trackingNumber` - Download TH waybill PDF
- **POST** `/api/shipping/download-bulk-pdf` - Download multiple waybill PDFs as ZIP
- **POST** `/api/shipping/download-bulk-th-pdf` - Download multiple TH waybill PDFs as merged PDF

---

## 11. Shipping - Pincode APIs

- **GET** `/api/shipping/check-pincode/:pincode` - Check pincode service availability (single)
- **POST** `/api/shipping/check-bulk-pincodes` - Check pincode service availability (bulk)

---

## 12. Database Analytics APIs

- **GET** `/api/net_profit_daily` - Get last N days net profit
- **GET** `/api/sales_unitCost_by_hour` - Get sales and unit cost by hour/date
- **POST** `/api/historical_stats_by_date` - Get historical statistics by date

---

## 13. User Management APIs

### User CRUD

- **POST** `/api/users` - Create new user (requires manager/admin/super_admin role)
- **GET** `/api/users` - Get all users (requires admin/super_admin role)
- **GET** `/api/users/uid/:uid` - Get user by UID (requires admin/super_admin role)
- **PUT** `/api/users/:userId/role` - Update user role and permissions (requires admin/super_admin role)
- **DELETE** `/api/users/:userId` - Delete user (requires admin/super_admin role)

### User Authentication & Verification

- **GET** `/api/user/role` - Get current user's role and permissions
- **POST** `/api/users/verify-email` - Verify user by email (requires admin/super_admin role)

---

## 14. Procurement APIs

### Product Management

- **POST** `/api/procurement/products` - Create a new product
- **GET** `/api/procurement/products` - Get all products (with pagination)
- **GET** `/api/procurement/products/search` - Search products
- **GET** `/api/procurement/products/:productId` - Get product by ID
- **PUT** `/api/procurement/products/:productId` - Update product
- **DELETE** `/api/procurement/products/:productId` - Delete product
- **PUT** `/api/procurement/products/:productId/status` - Update product status

### Variant Management

- **DELETE** `/api/procurement/variants/:variantId` - Delete variant

### Image Management

- **POST** `/api/procurement/products/:productId/images/upload-urls` - Generate image upload URLs
- **POST** `/api/procurement/images/:imageId/confirm` - Confirm image upload
- **GET** `/api/procurement/images/:imageId/view` - Get secure image view URL
- **DELETE** `/api/procurement/images/:imageId` - Delete image
- **DELETE** `/api/procurement/images/cleanup-pending` - Cleanup pending images
- **DELETE** `/api/procurement/images/cleanup-orphaned` - Cleanup orphaned images from Azure

### Storage Management

- **GET** `/api/procurement/storage/stats` - Get storage statistics

### Vendor Management (Product-specific)

- **POST** `/api/procurement/products/:productId/vendors` - Create vendor for product
- **GET** `/api/procurement/products/:productId/vendors` - Get vendors by product ID
- **GET** `/api/procurement/vendors/:vendorId` - Get vendor by ID
- **PUT** `/api/procurement/vendors/:vendorId` - Update vendor
- **DELETE** `/api/procurement/vendors/:vendorId` - Delete vendor
- **GET** `/api/procurement/vendors/search` - Search vendors

---

## 15. Master Vendor APIs

- **POST** `/api/masters/vendor/` - Create a new vendor (requires manager+ role)
- **GET** `/api/masters/vendor/` - Get all vendors with pagination (requires manager+ role)
- **GET** `/api/masters/vendor/:vendorId` - Get vendor by ID (requires manager+ role)
- **PUT** `/api/masters/vendor/:vendorId` - Update vendor (requires manager+ role)
- **DELETE** `/api/masters/vendor/:vendorId` - Delete vendor (requires manager+ role)

---

## 16. Master Product APIs

- **POST** `/api/masters/product/` - Create a new product (requires manager+ role)
- **GET** `/api/masters/product/` - Get all products with pagination (requires manager+ role)
- **GET** `/api/masters/product/:productId` - Get product by ID (requires manager+ role)
- **PUT** `/api/masters/product/:productId` - Update product (requires manager+ role)
- **DELETE** `/api/masters/product/:productId` - Delete product (requires manager+ role)

---

## 17. Receiving - Purchase Request APIs

- **POST** `/api/receiving/purchase-request/` - Create a new purchase request (requires manager+ role)
- **GET** `/api/receiving/purchase-request/` - Get all purchase requests with pagination (requires manager+ role)
- **GET** `/api/receiving/purchase-request/:requestId` - Get purchase request by ID (requires manager+ role)
- **PUT** `/api/receiving/purchase-request/:requestId` - Update purchase request (requires manager+ role)
- **DELETE** `/api/receiving/purchase-request/:requestId` - Delete purchase request (requires manager+ role)
- **PATCH** `/api/receiving/purchase-request/:requestId/status` - Update purchase request status (requires manager+ role)

---

## 18. Receiving - Quality Check APIs

### Quality Check Management

- **POST** `/api/receiving/quality-check/bulk` - Bulk create/update quality checks (requires manager+ role)
- **GET** `/api/receiving/quality-check/request/:requestId` - Get quality checks by request ID (requires manager+ role)
- **GET** `/api/receiving/quality-check/:qualityCheckId` - Get quality check by ID (requires manager+ role)
- **DELETE** `/api/receiving/quality-check/:qualityCheckId` - Delete quality check (requires manager+ role)

### Document Management

- **POST** `/api/receiving/quality-check/:requestId/documents` - Upload document (requires manager+ role)
- **GET** `/api/receiving/quality-check/:requestId/documents` - List documents (requires manager+ role)
- **DELETE** `/api/receiving/quality-check/documents/:documentId` - Delete document (requires manager+ role)
- **GET** `/api/receiving/quality-check/documents/:documentId/content` - Stream document content (requires manager+ role)

---

## 19. Stock Management APIs

### Inventory Management

- **GET** `/api/stock-management/variants/:variantId/inventory` - Get inventory for a specific variant (requires manager+ role)
- **GET** `/api/stock-management/variants/inventory` - Get inventory for all variants with pagination (requires manager+ role)

### Returns Management

- **GET** `/api/stock-management/returns/pending` - Get pending returns for approval (requires manager+ role)
- **GET** `/api/stock-management/returns` - Get all returns with optional status filter (requires manager+ role)
- **POST** `/api/stock-management/returns/:eventId/approve` - Approve a return (requires manager+ role)
- **POST** `/api/stock-management/returns/:eventId/reject` - Reject a return (requires manager+ role)

---

## Summary Statistics

### Total API Count by Category:

1. **Server Health & Metrics**: 3 APIs
2. **Product Metrics**: 4 APIs
3. **Core Analytics**: 13 APIs
4. **Geographic Analytics**: 3 APIs
5. **Source Analytics**: 8 APIs
6. **Amazon Entity Report**: 1 API
7. **Content Generation**: 2 APIs
8. **Customer Orders**: 8 APIs
9. **Shipping - Waybill**: 9 APIs
10. **Shipping - Download**: 4 APIs
11. **Shipping - Pincode**: 2 APIs
12. **Database Analytics**: 3 APIs
13. **User Management**: 7 APIs
14. **Procurement**: 17 APIs
15. **Master Vendor**: 5 APIs
16. **Master Product**: 5 APIs
17. **Receiving - Purchase Request**: 6 APIs
18. **Receiving - Quality Check**: 8 APIs
19. **Stock Management**: 6 APIs

### **TOTAL: 113 APIs**

---

## Authentication & Authorization

- **All `/api/*` routes require authentication** via Firebase Auth
- **Role-based access control** applies to:
  - User Management APIs (admin/super_admin)
  - Master Vendor/Product APIs (manager+)
  - Receiving APIs (manager+)
  - Stock Management APIs (manager+)
  - Procurement APIs (authenticated users)

---

## Rate Limiting

- Bulk upload endpoints have stricter rate limits
- Customer query endpoints have specific rate limits
- General API endpoints have standard rate limits
- Auth endpoints have more lenient rate limits

---

_Last Updated: Generated automatically from route files_

"""Policy layer for RBAC, rate limiting, and security."""
from typing import Optional, Dict, Any
from collections import defaultdict
from datetime import datetime, timedelta
from app.config.logging_config import get_logger
import time

logger = get_logger(__name__)


class PolicyLayer:
    """Manages access policies, rate limiting, and security."""
    
    def __init__(self):
        """Initialize policy layer."""
        logger.info("🔧 [POLICY] Initializing Policy Layer")
        self.rate_limits: Dict[str, list] = defaultdict(list)
        self.user_permissions: Dict[str, list] = defaultdict(list)
        
        # Default permissions
        self.default_permissions = ['analytics.query', 'meta.help']
        logger.info(f"✅ [POLICY] Policy Layer initialized | default_permissions={self.default_permissions}")
    
    def check_rate_limit(self, user_id: Optional[str], limit: int = 60, window: int = 60) -> bool:
        """Check if user is within rate limits."""
        if not user_id:
            user_id = 'anonymous'
        
        now = time.time()
        user_requests = self.rate_limits[user_id]
        
        # Remove old requests outside window
        user_requests[:] = [req_time for req_time in user_requests if now - req_time < window]
        
        # Check limit
        if len(user_requests) >= limit:
            return False
        
        # Record this request
        user_requests.append(now)
        return True
    
    def check_permissions(self, user_id: Optional[str], permission: str) -> bool:
        """Check if user has permission."""
        if not user_id:
            # Anonymous users get default permissions
            return permission in self.default_permissions
        
        user_perms = self.user_permissions.get(user_id, self.default_permissions)
        return permission in user_perms
    
    def mask_pii(self, data: Any, user_role: Optional[str] = None) -> Any:
        """Mask PII data unless user has elevated role."""
        # In production, implement proper PII detection and masking
        if user_role in ['admin', 'data_analyst']:
            return data
        
        # Basic PII masking
        if isinstance(data, dict):
            masked = {}
            pii_fields = ['email', 'phone', 'billing_phone', 'shipping_phone']
            for key, value in data.items():
                if any(pii_field in key.lower() for pii_field in pii_fields):
                    masked[key] = '***MASKED***'
                else:
                    masked[key] = self.mask_pii(value, user_role)
            return masked
        
        return data
    
    def apply_rls(self, user_id: Optional[str], query: str) -> str:
        """Apply row-level security filters."""
        # In production, inject RLS predicates based on user claims
        # For now, return query as-is
        return query


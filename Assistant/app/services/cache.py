"""Caching service for query results."""
from typing import Optional, Any
import hashlib
import json
import redis
from app.config.settings import settings
from app.models.schemas import QueryResponse
from app.config.logging_config import get_logger

logger = get_logger(__name__)


class CacheService:
    """Manages caching for query results."""
    
    def __init__(self):
        """Initialize cache service."""
        # Disable caching for testing - set to False to enable
        self.enabled = False
        
        if not self.enabled:
            logger.info("🚫 [CACHE] Caching disabled for testing")
            self.redis_client = None
            return
        
        logger.info(f"🔧 [CACHE] Initializing Cache Service | redis_url={settings.redis_url}")
        try:
            self.redis_client = redis.from_url(settings.redis_url, decode_responses=True)
            # Test connection
            self.redis_client.ping()
            logger.info("✅ [CACHE] Cache Service initialized successfully")
        except Exception as e:
            logger.warning(f"⚠️ [CACHE] Redis connection failed, caching disabled | error={str(e)}")
            self.redis_client = None
    
    def get_cache_key(self, query: str, user_id: Optional[str] = None) -> str:
        """Generate cache key from query and user."""
        key_data = {
            'query': query.lower().strip(),
            'user_id': user_id or 'anonymous'
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return f"query:{hashlib.md5(key_string.encode()).hexdigest()}"
    
    async def get(self, key: str) -> Optional[QueryResponse]:
        """Get cached response."""
        if not self.enabled or not self.redis_client:
            return None
        
        try:
            cached_data = self.redis_client.get(key)
            if cached_data:
                data = json.loads(cached_data)
                return QueryResponse(**data)
        except Exception:
            pass
        
        return None
    
    async def set(self, key: str, response: QueryResponse, ttl: int = 120):
        """Cache response."""
        if not self.enabled or not self.redis_client:
            return
        
        try:
            # Convert response to dict, excluding non-serializable fields
            data = response.dict()
            self.redis_client.setex(key, ttl, json.dumps(data))
        except Exception:
            pass


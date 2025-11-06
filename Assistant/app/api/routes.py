"""API routes for the agentic assistant."""
from fastapi import APIRouter, HTTPException, Depends, Header
from typing import Optional
from app.models.schemas import QueryRequest, QueryResponse
from app.services.orchestrator_v2 import OrchestratorV2
from app.services.policy import PolicyLayer
from app.services.cache import CacheService
from app.config.logging_config import get_logger
import time

logger = get_logger(__name__)

router = APIRouter()
orchestrator = OrchestratorV2()  # Use new LangGraph-based orchestrator
policy_layer = PolicyLayer()
cache_service = CacheService()


async def get_user_id(authorization: Optional[str] = Header(None)) -> Optional[str]:
    """Extract user ID from authorization header."""
    # In production, validate JWT and extract user_id
    if authorization:
        # Simplified - in production, decode JWT
        return authorization.split(' ')[-1] if ' ' in authorization else None
    return None


@router.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    user_id: Optional[str] = Depends(get_user_id)
):
    """Process analytics query."""
    start_time = time.time()
    
    logger.info(
        f"🔵 [API] Query request received | "
        f"query='{request.query[:100]}...' | "
        f"user_id={user_id} | "
        f"session_id={request.session_id}"
    )
    
    # Check cache (disabled for testing)
    cache_key = cache_service.get_cache_key(request.query, user_id)
    logger.debug(f"🔍 [API] Cache key generated: {cache_key} (caching disabled)")
    
    cached_response = await cache_service.get(cache_key)
    if cached_response:
        logger.info(f"✅ [API] Cache hit | cache_key={cache_key}")
        return cached_response
    
    logger.debug(f"❌ [API] Cache miss (caching disabled for testing) | cache_key={cache_key}")
    
    # Check rate limits
    rate_limit_check = policy_layer.check_rate_limit(user_id)
    logger.debug(f"⚡ [API] Rate limit check | user_id={user_id} | allowed={rate_limit_check}")
    if not rate_limit_check:
        logger.warning(f"⚠️ [API] Rate limit exceeded | user_id={user_id}")
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    # Check permissions
    permission_check = policy_layer.check_permissions(user_id, "analytics.query")
    logger.debug(f"🔐 [API] Permission check | user_id={user_id} | allowed={permission_check}")
    if not permission_check:
        logger.warning(f"⚠️ [API] Permission denied | user_id={user_id}")
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    # Process query
    logger.info(f"🚀 [API] Starting query processing | query='{request.query[:100]}...'")
    try:
        response = await orchestrator.process_query(
            query=request.query,
            user_id=user_id,
            session_id=request.session_id
        )
        logger.info(f"✅ [API] Query processing completed | request_id={response.request_id}")
    except Exception as e:
        logger.error(f"❌ [API] Query processing failed | error={str(e)}", exc_info=True)
        raise
    
    # Cache response
    logger.debug(f"💾 [API] Caching response | cache_key={cache_key}")
    await cache_service.set(cache_key, response, ttl=120)
    
    # Log execution time
    execution_time = time.time() - start_time
    response.metadata = response.metadata or {}
    response.metadata['execution_time_ms'] = execution_time * 1000
    
    logger.info(
        f"🏁 [API] Query completed | "
        f"request_id={response.request_id} | "
        f"execution_time_ms={execution_time * 1000:.2f} | "
        f"answer_length={len(response.answer)}"
    )
    
    return response


@router.get("/health")
async def health():
    """Health check endpoint."""
    logger.debug("💚 [API] Health check requested")
    return {"status": "healthy", "service": "agentic-assistant"}


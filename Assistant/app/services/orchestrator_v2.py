"""Orchestrator V2 - LangGraph-based query processing."""
from typing import Dict, Any, Optional
import uuid
from datetime import datetime
from app.graph.query_graph import create_query_graph, QueryState
from app.models.schemas import QueryResponse
from app.config.logging_config import get_logger
import traceback

logger = get_logger(__name__)


class OrchestratorV2:
    """LangGraph-based orchestrator for query processing."""
    
    def __init__(self):
        """Initialize orchestrator with LangGraph."""
        logger.info("🔧 [ORCHESTRATOR_V2] Initializing Orchestrator V2")
        self.graph = create_query_graph()
        logger.info("✅ [ORCHESTRATOR_V2] Orchestrator V2 initialized")
    
    async def process_query(
        self,
        query: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> QueryResponse:
        """
        Process user query using LangGraph.
        
        Args:
            query: User query string
            user_id: Optional user identifier
            session_id: Optional session identifier
            
        Returns:
            QueryResponse with answer and data
        """
        request_id = str(uuid.uuid4())
        
        logger.info(
            f"🎯 [ORCHESTRATOR_V2] Starting query processing | "
            f"request_id={request_id} | "
            f"query='{query[:100]}...' | "
            f"user_id={user_id} | session_id={session_id}"
        )
        
        try:
            # Create initial state
            initial_state: QueryState = {
                "query": query,
                "user_id": user_id,
                "session_id": session_id,
                "request_id": request_id,
                "task_spec": None,
                "plan": None,
                "step_results": {},
                "execution_results": {},
                "response": None,
                "error": None,
                "execution_trace": ""
            }
            
            # Execute graph
            final_state = await self.graph.ainvoke(initial_state)
            
            # Check for errors
            if final_state.get("error"):
                logger.error(
                    f"❌ [ORCHESTRATOR_V2] Query processing failed | "
                    f"request_id={request_id} | error={final_state['error']}"
                )
                return self._create_error_response(
                    request_id,
                    final_state["error"]
                )
            
            # Get response
            response = final_state.get("response")
            if response:
                response.request_id = request_id
                logger.info(
                    f"🎉 [ORCHESTRATOR_V2] Query processing completed successfully | "
                    f"request_id={request_id} | answer_length={len(response.answer)}"
                )
                return response
            else:
                logger.error(
                    f"❌ [ORCHESTRATOR_V2] No response generated | request_id={request_id}"
                )
                return self._create_error_response(
                    request_id,
                    "No response generated"
                )
                
        except Exception as e:
            logger.error(
                f"💥 [ORCHESTRATOR_V2] Query processing failed | "
                f"request_id={request_id} | error={str(e)}",
                exc_info=True
            )
            return self._create_error_response(
                request_id,
                f"Error processing query: {str(e)}",
                traceback=traceback.format_exc()
            )
    
    def _create_error_response(
        self,
        request_id: str,
        error_message: str,
        traceback: Optional[str] = None
    ) -> QueryResponse:
        """Create error response."""
        return QueryResponse(
            answer=f"I encountered an error: {error_message}",
            data=None,
            table=None,
            chart_spec=None,
            reasoning_trace=traceback or error_message,
            request_id=request_id,
            timestamp=datetime.utcnow()
        )


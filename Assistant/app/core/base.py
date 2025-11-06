"""Base classes for modular agents."""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, TypeVar, Generic, Type, Union
from pydantic import BaseModel, Field
from enum import Enum
from app.config.logging_config import get_logger

logger = get_logger(__name__)

# Type variables for agent I/O
TInput = TypeVar('TInput', bound=BaseModel)
TOutput = TypeVar('TOutput', bound=BaseModel)


class AgentType(str, Enum):
    """Types of agents in the system."""
    ROUTER = "router"
    PLANNER = "planner"
    GUARDRAIL = "guardrail"
    DATA_ACCESS = "data_access"
    COMPUTATION = "computation"
    COMPOSER = "composer"
    CUSTOM = "custom"


class AgentMetadata(BaseModel):
    """Metadata for agent registration."""
    agent_id: str
    agent_type: AgentType
    name: str
    description: str
    version: str = "1.0.0"
    dependencies: List[str] = Field(default_factory=list)
    capabilities: List[str] = Field(default_factory=list)
    enabled: bool = True

    model_config = {"extra": "allow"}


class BaseAgent(ABC, Generic[TInput, TOutput]):
    """Base class for all agents in the system with typed I/O support."""
    
    def __init__(
        self, 
        metadata: AgentMetadata,
        input_schema: Optional[Type[TInput]] = None,
        output_schema: Optional[Type[TOutput]] = None
    ):
        """
        Initialize agent with metadata and optional typed I/O schemas.
        
        Args:
            metadata: Agent metadata
            input_schema: Optional Pydantic model for input validation
            output_schema: Optional Pydantic model for output validation
        """
        self.metadata = metadata
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.logger = get_logger(f"{__name__}.{metadata.agent_id}")
        self.logger.info(
            f"🔧 [{metadata.agent_id.upper()}] Initializing agent | "
            f"type={metadata.agent_type.value} | "
            f"version={metadata.version} | "
            f"typed_io={input_schema is not None and output_schema is not None}"
        )
    
    def validate_inputs(self, inputs: Union[Dict[str, Any], TInput]) -> TInput:
        """
        Validate and parse inputs using Pydantic schema if available.
        
        Args:
            inputs: Input data (dict or already validated model)
            
        Returns:
            Validated input model or original dict if no schema
        """
        if self.input_schema is None:
            # No schema defined, return as-is (backward compatibility)
            return inputs  # type: ignore
        
        if isinstance(inputs, self.input_schema):
            # Already validated
            return inputs
        
        if isinstance(inputs, dict):
            # Validate dict against schema
            try:
                return self.input_schema(**inputs)
            except Exception as e:
                self.logger.error(f"❌ Input validation failed | error={str(e)} | inputs={inputs}")
                raise ValueError(f"Invalid inputs: {str(e)}")
        
        # Fallback
        return inputs  # type: ignore
    
    def _log_inputs(self, inputs: Union[Dict[str, Any], TInput], context: Optional[Dict[str, Any]] = None) -> None:
        """Log agent inputs for debugging."""
        import json
        
        try:
            # Convert inputs to dict if it's a Pydantic model
            if hasattr(inputs, 'model_dump'):
                inputs_dict = inputs.model_dump()
            elif hasattr(inputs, 'dict'):
                inputs_dict = inputs.dict()
            elif isinstance(inputs, dict):
                inputs_dict = inputs
            else:
                inputs_dict = {"raw": str(inputs)[:500]}  # Truncate if too long
            
            # Truncate large data structures
            inputs_dict = self._truncate_for_logging(inputs_dict)
            
            self.logger.info(
                f"📥 [{self.metadata.agent_id.upper()}] Input | "
                f"inputs={json.dumps(inputs_dict, default=str, indent=2)[:2000]} | "
                f"context={json.dumps(context, default=str)[:500] if context else None}"
            )
        except Exception as e:
            self.logger.warning(f"📥 [{self.metadata.agent_id.upper()}] Input logging failed | error={str(e)}")
    
    def _log_outputs(self, outputs: Union[Dict[str, Any], TOutput], execution_time_ms: Optional[float] = None) -> None:
        """Log agent outputs for debugging."""
        import json
        
        try:
            # Convert outputs to dict if it's a Pydantic model
            if hasattr(outputs, 'model_dump'):
                outputs_dict = outputs.model_dump()
            elif hasattr(outputs, 'dict'):
                outputs_dict = outputs.dict()
            elif isinstance(outputs, dict):
                outputs_dict = outputs
            else:
                outputs_dict = {"raw": str(outputs)[:500]}
            
            # Truncate large data structures
            outputs_dict = self._truncate_for_logging(outputs_dict)
            
            log_msg = f"📤 [{self.metadata.agent_id.upper()}] Output | outputs={json.dumps(outputs_dict, default=str, indent=2)[:2000]}"
            if execution_time_ms:
                log_msg += f" | execution_time_ms={execution_time_ms:.2f}"
            
            self.logger.info(log_msg)
        except Exception as e:
            self.logger.warning(f"📤 [{self.metadata.agent_id.upper()}] Output logging failed | error={str(e)}")
    
    def _truncate_for_logging(self, data: Any, max_length: int = 1000, max_items: int = 10) -> Any:
        """Truncate large data structures for logging."""
        import json
        
        if isinstance(data, dict):
            truncated = {}
            for i, (key, value) in enumerate(data.items()):
                if i >= max_items:
                    truncated[f"... ({len(data) - max_items} more keys)"] = None
                    break
                truncated[key] = self._truncate_for_logging(value, max_length, max_items)
            return truncated
        elif isinstance(data, list):
            if len(data) > max_items:
                return [
                    self._truncate_for_logging(item, max_length, max_items) 
                    for item in data[:max_items]
                ] + [f"... ({len(data) - max_items} more items)"]
            return [self._truncate_for_logging(item, max_length, max_items) for item in data]
        elif isinstance(data, str):
            if len(data) > max_length:
                return data[:max_length] + "... (truncated)"
            return data
        else:
            data_str = str(data)
            if len(data_str) > max_length:
                return data_str[:max_length] + "... (truncated)"
            return data
    
    @abstractmethod
    async def execute(
        self, 
        inputs: Union[Dict[str, Any], TInput], 
        context: Optional[Dict[str, Any]] = None
    ) -> Union[Dict[str, Any], TOutput]:
        """
        Execute the agent's main logic.
        
        Args:
            inputs: Input data for the agent (dict or validated model)
            context: Optional context information
            
        Returns:
            Agent execution results (dict or validated model)
        """
        pass
    
    def get_metadata(self) -> AgentMetadata:
        """Get agent metadata."""
        return self.metadata
    
    def is_enabled(self) -> bool:
        """Check if agent is enabled."""
        return self.metadata.enabled
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.metadata.agent_id}, type={self.metadata.agent_type.value})"


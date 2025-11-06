# Modular Architecture with LangChain and LangGraph

This document describes the new modular architecture that uses LangChain and LangGraph for building plug-and-play agents.

## Overview

The system has been refactored to use:
- **LangChain**: For agent and tool abstraction
- **LangGraph**: For state-based orchestration
- **Modular Design**: Plug-and-play agents that can be easily added/removed

## Architecture Components

### 1. Core Module (`app/core/`)

Base classes and interfaces for all agents:

- **BaseAgent**: Abstract base class for all agents
- **AgentMetadata**: Metadata schema for agent registration
- **AgentRegistry**: Singleton registry for managing agents

### 2. Agents V2 (`app/agents_v2/`)

LangChain-based agents that extend `BaseAgent`:

- **RouterAgentV2**: Intent classification and slot extraction
- **PlannerAgentV2**: Execution plan generation
- **ComposerAgentV2**: Natural language answer composition

### 3. Tools (`app/tools/`)

LangChain tools for data access and computation:

- **Data Access Tools**: `create_sales_db_tool`, `create_amazon_ads_db_tool`, `create_meta_ads_db_tool`
- **Computation Tools**: `create_compute_metric_tool`, `create_join_data_tool`, `create_aggregate_data_tool`

### 4. Graph (`app/graph/`)

LangGraph state graph for orchestration:

- **QueryState**: TypedDict for state management
- **create_query_graph()**: Factory function that creates the compiled graph

### 5. Orchestrator V2 (`app/services/orchestrator_v2.py`)

New orchestrator that uses LangGraph instead of manual step-by-step execution.

## Key Benefits

### 1. Modularity
- Agents can be added/removed without affecting the entire system
- Each agent is independent and can be tested in isolation
- Clear separation of concerns

### 2. Flexibility
- Easy to add new agents by extending `BaseAgent`
- Tools can be registered dynamically
- State graph can be modified without changing core logic

### 3. Maintainability
- LangGraph provides visual workflow representation
- State management is explicit and typed
- Error handling is built into the graph structure

## Adding a New Agent

### Step 1: Create Agent Class

```python
from app.core.base import BaseAgent, AgentMetadata, AgentType
from typing import Dict, Any, Optional

class MyCustomAgent(BaseAgent):
    def __init__(self):
        metadata = AgentMetadata(
            agent_id="my_custom_agent",
            agent_type=AgentType.CUSTOM,
            name="My Custom Agent",
            description="Does something custom",
            version="1.0.0",
            capabilities=["custom_capability"]
        )
        super().__init__(metadata)
        # Initialize your agent here
    
    async def execute(self, inputs: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        # Implement your agent logic
        return {"success": True, "result": "..."}
```

### Step 2: Register Agent

```python
from app.core.registry import agent_registry
from app.agents_v2.my_custom_agent import MyCustomAgent

# Register agent
agent = MyCustomAgent()
agent_registry.register(agent)
```

### Step 3: Add to Graph (if needed)

If your agent should be part of the query processing pipeline, add it to the graph in `app/graph/query_graph.py`:

```python
workflow.add_node("my_custom_agent", my_custom_agent_node)
workflow.add_edge("previous_node", "my_custom_agent")
workflow.add_edge("my_custom_agent", "next_node")
```

## Adding a New Tool

### Step 1: Create Tool

```python
from langchain_core.tools import tool
from pydantic import BaseModel, Field

class MyToolInput(BaseModel):
    param1: str = Field(description="Description of param1")

@tool(args_schema=MyToolInput)
def my_custom_tool(param1: str) -> Dict[str, Any]:
    """Tool description."""
    # Implement tool logic
    return {"result": "..."}
```

### Step 2: Export Tool

Add to `app/tools/__init__.py`:

```python
from app.tools.my_custom_tool import my_custom_tool

__all__ = [..., "my_custom_tool"]
```

## Graph Flow

The query processing flow follows this graph:

```
router → validate_task → planner → validate_plan → execute_plan → composer → END
```

Each node can:
- Modify the state
- Return to continue flow
- Return error to end flow

## State Management

The `QueryState` TypedDict contains:
- `query`: User query string
- `task_spec`: Parsed task specification
- `plan`: Execution plan
- `step_results`: Results from each execution step
- `execution_results`: Final aggregated results
- `response`: Final query response
- `error`: Error message if any

## Migration from Old Architecture

The old agents (`app/agents/`) are still available for backward compatibility. To migrate:

1. Replace `Orchestrator` with `OrchestratorV2` in your code
2. Update imports to use `agents_v2` instead of `agents`
3. Ensure all async operations use `await`

## Testing

Each agent can be tested independently:

```python
async def test_my_agent():
    agent = MyCustomAgent()
    result = await agent.execute({"input": "test"})
    assert result["success"] == True
```

## Best Practices

1. **Always extend BaseAgent**: Don't create agents from scratch
2. **Use AgentMetadata**: Always provide metadata for registration
3. **Handle errors gracefully**: Return error dicts instead of raising exceptions
4. **Log appropriately**: Use the logger from BaseAgent
5. **Type hints**: Use proper type hints for all methods

## Future Enhancements

- Agent versioning and A/B testing
- Dynamic graph construction based on query type
- Agent performance monitoring
- Automatic agent discovery from configuration files


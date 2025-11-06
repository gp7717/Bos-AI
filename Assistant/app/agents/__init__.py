"""Agent implementations - Legacy agents still in use."""

# These agents are still used by the new LangGraph-based system:
# - GuardrailAgent: Validation and safety checks
# - DataAccessAgent classes: Database access (used via get_data_access_agent)
# - ComputationAgent: Metric calculations and aggregations

# Note: Router, Planner, and Composer agents have been replaced by V2 versions
# in app/agents_v2/ which use LangChain

__all__ = [
    "GuardrailAgent",
    "ComputationAgent",
    "get_data_access_agent",
]

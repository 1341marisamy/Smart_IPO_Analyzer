from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    """
    The state for the Smart IPO Advisor multi-agent system.
    """
    messages: Annotated[list[AnyMessage], add_messages]
    ipo_name: str
    pdf_findings: str
    web_findings: str
    final_verdict: str

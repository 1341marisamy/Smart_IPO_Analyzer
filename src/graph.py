from langgraph.graph import StateGraph, END
from src.state import AgentState
from src.agents import (
    supervisor_node,
    market_scraper_node,
    rag_analyst_node,
    risk_auditor_node
)

def router(state: AgentState) -> str:
    """
    Looks at the last message from the Supervisor to determine the next node.
    """
    last_message = state["messages"][-1]
    # The supervisor node returns a message like "Supervisor routes to: Market_Scraper"
    content = last_message.content
    
    if "Market_Scraper" in content:
        return "Market_Scraper"
    elif "RAG_Analyst" in content:
        return "RAG_Analyst"
    elif "Risk_Auditor" in content:
        return "Risk_Auditor"
    else:
        return "FINISH"

# Initialize the StateGraph
workflow = StateGraph(AgentState)

# Add all the nodes
workflow.add_node("Supervisor", supervisor_node)
workflow.add_node("Market_Scraper", market_scraper_node)
workflow.add_node("RAG_Analyst", rag_analyst_node)
workflow.add_node("Risk_Auditor", risk_auditor_node)

# Set the entry point to Supervisor
workflow.set_entry_point("Supervisor")

# Add edges from Scraper and Analyst back to the Supervisor
workflow.add_edge("Market_Scraper", "Supervisor")
workflow.add_edge("RAG_Analyst", "Supervisor")

# Add conditional edges from Supervisor to the specialized nodes
workflow.add_conditional_edges(
    "Supervisor",
    router,
    {
        "Market_Scraper": "Market_Scraper",
        "RAG_Analyst": "RAG_Analyst",
        "Risk_Auditor": "Risk_Auditor",
        "FINISH": END
    }
)

# Add an edge from Auditor to END
workflow.add_edge("Risk_Auditor", END)

# Compile the graph
app = workflow.compile()

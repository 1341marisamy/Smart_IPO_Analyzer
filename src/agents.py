from typing import Dict, Any, Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from src.state import AgentState
from src.tools import search_web_sentiment, search_ipo_pdf
from pydantic import BaseModel, Field

# Base LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# ---------------------------------------------------------------------
# 1. Supervisor Agent
# ---------------------------------------------------------------------
class SupervisorDecision(BaseModel):
    next_action: Literal["Market_Scraper", "RAG_Analyst", "Risk_Auditor", "FINISH"] = Field(
        description="The next agent to route to."
    )

supervisor_prompt = """You are the Supervisor of the Smart IPO Advisor.
Your job is to manage the workflow to evaluate the {ipo_name} IPO.
You have three agents at your disposal:
1. 'Market_Scraper': Use this if you need live market sentiment and GMP (Grey Market Premium).
2. 'RAG_Analyst': Use this if you need deep financial findings (debt, revenue, risks) from the uploaded PDF.
3. 'Risk_Auditor': Use this to compile the final 'Subscribe or Avoid' report ONLY WHEN BOTH web findings and PDF findings are comprehensively gathered.

Current Data Status:
- Web Findings Gathered: {web_status}
- PDF Findings Gathered: {pdf_status}

Analyze the data status. If web findings are missing, route to Market_Scraper. 
If PDF findings are missing, route to RAG_Analyst. 
If both are collected sufficiently, route to Risk_Auditor.
If the final verdict is already made, choose FINISH.
"""

def supervisor_node(state: AgentState) -> Dict[str, Any]:
    ipo_name = state.get("ipo_name", "Unknown")
    web_findings = state.get("web_findings", "")
    pdf_findings = state.get("pdf_findings", "")
    
    web_status = "YES" if len(web_findings) > 10 else "NO"
    pdf_status = "YES" if len(pdf_findings) > 10 else "NO"
    
    sys_msg = SystemMessage(content=supervisor_prompt.format(
        ipo_name=ipo_name, 
        web_status=web_status, 
        pdf_status=pdf_status
    ))
    
    # Simple formatting of user request for supervisor context
    messages = [sys_msg] + state["messages"]

    # LLM decides next agent
    supervisor_llm = llm.with_structured_output(SupervisorDecision)
    decision = supervisor_llm.invoke(messages)
    
    # We will just return the decision as a message or handle it in the router.
    # To use this in a conditional edge, the router will look at the last message or we can set it in state.
    # But since state only expects messages, ipo_name, findings, etc., let's return a dummy message to help the router.
    return {"messages": [SystemMessage(content=f"Supervisor routes to: {decision.next_action}", name="Supervisor")]}


# ---------------------------------------------------------------------
# 2. Market_Scraper Agent
# ---------------------------------------------------------------------
scraper_prompt = """You are the Market Scraper agent. Your task is to extract real-time market sentiment and Grey Market Premium (GMP) for the {ipo_name} IPO.
Use your tool 'search_web_sentiment' to search the web, then compile a neat summary of the GMP and news sentiment."""

# Bind tool
scraper_llm = llm.bind_tools([search_web_sentiment])

def market_scraper_node(state: AgentState) -> Dict[str, Any]:
    ipo_name = state.get("ipo_name", "Unknown")
    sys_msg = SystemMessage(content=scraper_prompt.format(ipo_name=ipo_name))
    
    # Provide the last relevant message to the llm
    response = scraper_llm.invoke([sys_msg] + state["messages"])
    
    # If the LLM called a tool, run it
    if response.tool_calls:
        tool_call = response.tool_calls[0]
        # execute tool manually since it's a simple node
        tool_res = search_web_sentiment.invoke(tool_call["args"])
        
        # Second invocation to get final answer
        tool_msg = ToolMessage(content=tool_res, tool_call_id=tool_call["id"], name=tool_call["name"])
        final_response = llm.invoke([sys_msg] + state["messages"] + [response, tool_msg])
        web_info = final_response.content
    else:
        web_info = response.content
        
    return {
        "web_findings": web_info,
        "messages": [SystemMessage(content=f"Market Scraper gathered web findings.", name="Market_Scraper")]
    }

# ---------------------------------------------------------------------
# 3. RAG_Analyst Agent
# ---------------------------------------------------------------------
analyst_prompt = """You are the RAG Analyst agent. Your task is to extract deep financial findings (debt, revenue, risks, objects of the issue) from the uploaded PDF for the {ipo_name} IPO.
Use your tool 'search_ipo_pdf' to query the company financials, risk factors, and valuation. 
Run multiple queries if needed. Compile a neat summary of the PDF findings."""

analyst_llm = llm.bind_tools([search_ipo_pdf])

def rag_analyst_node(state: AgentState) -> Dict[str, Any]:
    ipo_name = state.get("ipo_name", "Unknown")
    sys_msg = SystemMessage(content=analyst_prompt.format(ipo_name=ipo_name))
    
    # First invocation
    response = analyst_llm.invoke([sys_msg, HumanMessage(content="Search for company debt, revenues, risk factors, and valuation in the PDF.")])
    
    if response.tool_calls:
        messages = [sys_msg] + state["messages"] + [response]
        for tool_call in response.tool_calls:
            # Execute tool
            tool_res = search_ipo_pdf.invoke(tool_call["args"])
            tool_msg = ToolMessage(content=tool_res, tool_call_id=tool_call["id"], name=tool_call["name"])
            messages.append(tool_msg)
            
        final_response = llm.invoke(messages)
        pdf_info = final_response.content
    else:
        pdf_info = response.content

    return {
        "pdf_findings": pdf_info,
        "messages": [SystemMessage(content=f"RAG Analyst gathered PDF findings.", name="RAG_Analyst")]
    }


# ---------------------------------------------------------------------
# 4. Risk_Auditor Agent
# ---------------------------------------------------------------------
auditor_prompt = """You are the Risk Auditor of the Smart IPO Advisor.
Your job is to read both the Web Findings and the PDF Findings for the {ipo_name} IPO, compare them, synthesize the data, and write a final comprehensive 'Subscribe or Avoid' report.

---
Web Findings (Market Sentiment & GMP):
{web_findings}

---
PDF Findings (Financials & Risk):
{pdf_findings}

---
Write a detailed, structured final verdict comparing both sources, noting risks, and giving a final 'Subscribe' or 'Avoid' recommendation with clear reasoning."""

def risk_auditor_node(state: AgentState) -> Dict[str, Any]:
    ipo_name = state.get("ipo_name", "Unknown")
    web_findings = state.get("web_findings", "No web findings available.")
    pdf_findings = state.get("pdf_findings", "No PDF findings available.")
    
    sys_msg = SystemMessage(content=auditor_prompt.format(
        ipo_name=ipo_name,
        web_findings=web_findings,
        pdf_findings=pdf_findings
    ))
    
    response = llm.invoke([sys_msg])
    verdict = response.content
    
    return {
        "final_verdict": verdict,
        "messages": [SystemMessage(content="Auditor has delivered the final verdict.", name="Risk_Auditor")]
    }

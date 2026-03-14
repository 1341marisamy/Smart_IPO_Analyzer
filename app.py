import streamlit as st
import os
import subprocess
from langchain_core.messages import HumanMessage
from src.graph import app as graph_app
from src.state import AgentState

def main():
    st.set_page_config(page_title="Smart IPO Advisor", page_icon="📈", layout="wide")
    st.title("📈 Smart IPO Advisor")
    st.markdown("Analyze Initial Public Offerings (IPOs) using live market sentiment and official financial documents.")
    
    # -----------------------------
    # Sidebar: Document Processing
    # -----------------------------
    with st.sidebar:
        st.header("1. Document Ingestion")
        st.write("Upload an IPO RHP (Red Herring Prospectus) in PDF format to build the knowledge base.")
        
        uploaded_file = st.file_uploader("Upload IPO RHP (PDF)", type=["pdf"])
        
        if uploaded_file is not None:
            # Create data dir if not exists
            data_dir = os.path.join(os.path.dirname(__file__), "data")
            os.makedirs(data_dir, exist_ok=True)
            
            # Save file
            file_path = os.path.join(data_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            st.success(f"File '{uploaded_file.name}' saved locally.")
            
            # Button to trigger generic ingestion.py script in background
            if st.button("Process Document (Ingest)"):
                with st.spinner("Embedding text and uploading to MongoDB Atlas..."):
                    try:
                        # Call ingestion as subprocess to keep UI clean and separate dependencies context if desired
                        result = subprocess.run(
                            ["python", "ingestion.py"], 
                            capture_output=True, 
                            text=True
                        )
                        if result.returncode == 0:
                            st.success("Ingestion Complete!")
                            with st.expander("Show Ingestion Logs"):
                                st.text(result.stdout)
                        else:
                            st.error("Ingestion failed!")
                            with st.expander("Show Error Logs"):
                                st.text(result.stderr)
                                st.text(result.stdout)
                    except Exception as e:
                        st.error(f"Execution Error: {str(e)}")
                        
        st.markdown("---")
        st.info("Make sure you have correctly configured your `.env` file with `OPENAI_API_KEY`, `TAVILY_API_KEY`, and `MONGO_URI`.")
                        
    # -----------------------------
    # Main Area: IPO Analysis Chat
    # -----------------------------
    st.header("2. Request Analysis")
    
    ipo_name = st.text_input("Enter the IPO Name (e.g., 'Tata Technologies', 'Reddit')", placeholder="Search IPO...")
    
    if st.button("Analyze IPO"):
        if not ipo_name.strip():
            st.warning("Please enter an IPO name to begin.")
        else:
            with st.spinner(f"Initiating multi-agent analysis for '{ipo_name}'..."):
                
                # Setup initial state
                initial_state: AgentState = {
                    "messages": [HumanMessage(content=f"Please analyze the {ipo_name} IPO.")],
                    "ipo_name": ipo_name,
                    "pdf_findings": "",
                    "web_findings": "",
                    "final_verdict": ""
                }
                
                output_container = st.container()
                
                try:
                    # Stream updates live
                    for s in graph_app.stream(initial_state, {"recursion_limit": 20}):
                        # s is a dict mapping node_name -> dict of state updates
                        for node, updates in s.items():
                            with output_container:
                                st.markdown(f"**Update from {node}:**")
                                
                                # Show agent action/messages
                                if "messages" in updates and updates["messages"]:
                                    msg = updates["messages"][-1].content
                                    st.write(msg)
                                    
                                # If Auditor delivered the final verdict, display it nicely
                                if "final_verdict" in updates and updates["final_verdict"]:
                                    st.success("🎯 Final Verdict Compiled!")
                                    st.markdown("### Subscribe or Avoid Report")
                                    st.info(updates["final_verdict"])
                                    
                                st.divider()
                                
                except Exception as e:
                    st.error(f"LangGraph execution encountered an error: {str(e)}")

if __name__ == "__main__":
    main()

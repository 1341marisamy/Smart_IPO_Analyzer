import os
from typing import List, Dict, Any
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from pymongo import MongoClient
from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv()

# Initialize embeddings
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=3072)

# MongoDB setup
mongo_uri = os.environ.get("MONGO_URI")
try:
    mongo_client = MongoClient(mongo_uri)
    db = mongo_client["ipo_db"]
    collection = db["ipo_vectors"]
except Exception as e:
    # Handle gracefully if no MongoDB URI is set or connection fails
    mongo_client = None
    db = None
    collection = None
    print(f"Warning: MongoDB connection failed or MONGO_URI is missing. Error: {e}")

# Tavily setup
tavily_api_key = os.environ.get("TAVILY_API_KEY")
tavily_client = TavilyClient(api_key=tavily_api_key) if tavily_api_key else None


@tool
def search_ipo_pdf(query: str) -> str:
    """
    Search the uploaded IPO PDF files for specific information (debt, revenue, risk factors).
    Connects to MongoDB Atlas Vector Search and returns the top relevant chunks.
    
    Args:
        query: The search query to find in the PDF.
        
    Returns:
        A string containing the concatenated text of the top 5 most relevant chunks.
    """
    if collection is None:
        return "Error: MongoDB is not connected. Unable to search the PDF."
        
    try:
        # Embed the query
        query_vector = embeddings_model.embed_query(query)
        
        # Define the Atlas Vector Search pipeline
        # Note: You must have an Atlas Vector Search index named 'default' configured on 'ipo_vectors'.
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "default",
                    "path": "embedding",
                    "queryVector": query_vector,
                    "numCandidates": 50,
                    "limit": 5
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "text": 1,
                    "source": 1,
                    "page": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]
        
        results = list(collection.aggregate(pipeline))
        
        if not results:
            return "No relevant information found in the uploaded PDF."
            
        formatted_results = []
        for doc in results:
            text = doc.get('text', '')
            source = doc.get('source', 'Unknown')
            page = doc.get('page', 'Unknown')
            formatted_results.append(f"[Source: {source}, Page: {page}]\n{text}")
            
        return "\n\n---\n\n".join(formatted_results)
        
    except Exception as e:
        return f"Error during PDF search: {str(e)}"

@tool
def search_web_sentiment(ipo_name: str) -> str:
    """
    Search the web for the current Grey Market Premium (GMP) and recent news/sentiment for the given IPO.
    
    Args:
        ipo_name: The name of the IPO to search for.
        
    Returns:
        A string containing the summary of the web search results.
    """
    if tavily_client is None:
        return "Error: Tavily API key is missing. Unable to perform web search."
        
    try:
        query = f"{ipo_name} IPO Grey Market Premium GMP current price expected listing gains and recent news sentiment"
        
        # Perform search using Tavily
        response = tavily_client.search(
            query=query,
            search_depth="advanced",
            include_answer=True,
            max_results=5
        )
        
        # Extract the AI-generated answer and top results
        answer = response.get("answer", "")
        results = response.get("results", [])
        
        output = []
        if answer:
            output.append(f"AI Summary:\n{answer}\n")
            
        output.append("Top Web Results:")
        for res in results:
            title = res.get('title', '')
            content = res.get('content', '')
            url = res.get('url', '')
            output.append(f"- {title}: {content} ({url})")
            
        return "\n".join(output)
        
    except Exception as e:
        return f"Error during web search: {str(e)}"

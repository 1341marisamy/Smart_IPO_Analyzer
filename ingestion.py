import os
import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

def ingest_pdfs():
    """
    Reads PDFs from the data/ folder, splits them into chunks, embeds them 
    using OpenAI, and pushes to MongoDB Atlas vector store.
    """
    print("Starting ingestion process...")
    
    # Setup MongoDB
    mongo_uri = os.environ.get("MONGO_URI")
    if not mongo_uri or "your_mongo_uri" in mongo_uri:
        print("Error: MONGO_URI is not set correctly in your .env file.")
        return
        
    try:
        client = MongoClient(mongo_uri)
        # Test connection
        client.admin.command('ping')
        db = client["Smart_IPO"]
        collection = db["embeddings"]
        print("Connected to MongoDB Atlas successfully.")
    except Exception as e:
        print(f"Failed to connect to MongoDB: {e}")
        return
    
    # Initialize OpenAI Embeddings
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=3072)
    except Exception as e:
        print(f"Failed to initialize OpenAI Embeddings: {e}")
        return
    
    # Locate data folder
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created {data_dir} directory. Please add PDFs and try again.")
        return
        
    pdf_files = glob.glob(os.path.join(data_dir, "*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {data_dir}.")
        return
        
    # Process each PDF
    for pdf_path in pdf_files:
        print(f"Loading {pdf_path}...")
        try:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            print(f"Loaded {len(docs)} pages from {os.path.basename(pdf_path)}.")
            
            # Splitter
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = splitter.split_documents(docs)
            print(f"Split document into {len(chunks)} chunks.")
            
            # Embed and insert in batches
            batch_size = 50
            records = []
            
            for i, chunk in enumerate(chunks):
                # We extract embedding and prepare python dictionary compatible with MongoDB
                embedding = embeddings.embed_query(chunk.page_content)
                record = {
                    "text": chunk.page_content,
                    "source": os.path.basename(pdf_path),
                    "page": chunk.metadata.get("page", 0),
                    "embedding": embedding
                }
                records.append(record)
                
                # Insert if batch is full
                if len(records) >= batch_size:
                    collection.insert_many(records)
                    print(f"Inserted {i+1}/{len(chunks)} chunks into MongoDB...")
                    records = []
                    
            # Insert any remaining chunks
            if records:
                collection.insert_many(records)
                print(f"Inserted final {len(records)} chunks into MongoDB.")
                
            print(f"Finished processing {os.path.basename(pdf_path)}.")
        
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
            
    print("Ingestion complete.")

if __name__ == "__main__":
    ingest_pdfs()

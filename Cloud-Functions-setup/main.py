# main.py

import os
import json
import psycopg2
from pgvector.psycopg2 import register_vector
from vertexai.language_models import TextEmbeddingModel, TextEmbedding
from vertexai.preview.generative_models import GenerativeModel, Tool, Part, FunctionDeclaration, GenerationResponse 
from datetime import datetime # Added datetime import for isoformat, if not already there
import numpy as np # Added numpy import for np.integer, if not already there

# FastAPI specific imports
from fastapi import FastAPI, Request, HTTPException 
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# IMPORTANT: Load environment variables from .env file for LOCAL testing
# from dotenv import load_dotenv
# load_dotenv() 

# --- Configuration (from Environment Variables) ---
DB_HOST = os.environ.get('DB_HOST') 
DB_NAME = os.environ.get('DB_NAME')
DB_USER = os.environ.get('DB_USER')
DB_PASSWORD = os.environ.get('DB_PASSWORD')
DB_PORT = os.environ.get('DB_PORT', '5432') 

EMBEDDING_MODEL_NAME = os.environ.get('EMBEDDING_MODEL_NAME', "text-embedding-004")
EMBEDDING_DIMENSION = int(os.environ.get('EMBEDDING_DIMENSION', "768"))
# GEMINI_MODEL_NAME = os.environ.get('GEMINI_MODEL_NAME', "gemini-1.5-flash-001")
GEMINI_MODEL_NAME = os.environ.get('GEMINI_MODEL_NAME', "gemini-2.0-flash")

# Global instances for warm starts
embedding_model_instance = None
generative_model_instance = None
_db_single_conn = None 

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Incident Semantic Search with Gemini 1.5 and PgVector",
    description="API for querying incident descriptions using semantic search powered by Vertex AI Gemini and PgVector.",
    version="1.0.0",
)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models for Request/Response ---
class QueryRequest(BaseModel):
    query_text: str

# --- Helper Functions (keep these exactly as they are) ---
def get_embedding_model():
    global embedding_model_instance
    if embedding_model_instance is None:
        try:
            embedding_model_instance = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL_NAME)
            print(f"Embedding model '{EMBEDDING_MODEL_NAME}' initialized.")
        except Exception as e:
            print(f"Failed to initialize embedding model: {e}")
            raise RuntimeError("Could not initialize embedding model.") from e
    return embedding_model_instance

# --- MODIFIED: Pass tools_to_register when getting the generative model ---
def get_generative_model(tools_to_register=None):
    global generative_model_instance
    if generative_model_instance is None:
        try:
            # Pass tools directly during model initialization
            generative_model_instance = GenerativeModel(GEMINI_MODEL_NAME, tools=tools_to_register)
            print(f"Generative model '{GEMINI_MODEL_NAME}' initialized with tools.")
        except Exception as e:
            print(f"Failed to initialize generative model: {e}")
            raise RuntimeError("Could not initialize generative model.") from e
    return generative_model_instance

def get_db_connection():
    global _db_single_conn
    
    # Check if connection exists and is still open. If not, create a new one.
    if _db_single_conn is None or _db_single_conn.closed:
        try:
            print(f"Attempting to establish NEW DB connection: Host={DB_HOST}, DB={DB_NAME}, User={DB_USER}, Port={DB_PORT}")
            conn = psycopg2.connect(
                host=DB_HOST,
                database=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD,
                port=DB_PORT,
                connect_timeout=10 # Added timeout for robustness
            )
            register_vector(conn) # Register vector type for this new connection
            print("New database connection established and pgvector type registered.")
            _db_single_conn = conn # Store this new connection globally
        except Exception as e:
            print(f"Failed to connect to database: {e}")
            raise RuntimeError("Could not establish a database connection.") from e
    else:
        # If connection already exists and is open, ensure vector type is registered (redundant but safe)
        try:
            register_vector(_db_single_conn) # Re-register for existing connection
            print("Reusing existing database connection and re-registered pgvector type.")
        except Exception as e:
            # This case implies something went wrong with the existing connection
            print(f"Failed to re-register pgvector for existing connection: {e}")
            # Consider closing _db_single_conn and forcing a new connection on next call
            if _db_single_conn:
                _db_single_conn.close()
                _db_single_conn = None
            raise RuntimeError("Issue with existing database connection.") from e

    return _db_single_conn

def get_gecko_embedding(texts: str | list[str]) -> list[float] | list[list[float]]:
    model = get_embedding_model()
    is_single_text = isinstance(texts, str)
    texts_to_embed = [texts] if is_single_text else texts
    if not texts_to_embed:
        return [] if not is_single_text else [0.0] * EMBEDDING_DIMENSION
    zero_vector = [0.0] * EMBEDDING_DIMENSION
    try:
        batch_embeddings_response: list[TextEmbedding] = model.get_embeddings(texts_to_embed)
        results = [emb.values for emb in batch_embeddings_response]
        if is_single_text:
            return results[0]
        else:
            return results
    except Exception as e:
        print(f"Error generating embeddings for text(s): '{texts_to_embed[0][:50]}...'. Error: {e}")
        if is_single_text:
            return zero_vector
        else:
            return [zero_vector] * len(texts_to_embed)

def fetch_similar_incidents_from_db(query_text: str) -> list[dict]:
    print(f"Tool call: Fetching similar incidents for query: '{query_text}'")
    query_embedding_values = get_gecko_embedding(query_text)

    if not query_embedding_values:
        print("Failed to generate embedding for the query. Returning empty results.")
        return []

    # --- DEBUGGING PRINTS ---
    print(f"DEBUG: Type of query_embedding_values BEFORE DB query: {type(query_embedding_values)}")
    if isinstance(query_embedding_values, list):
        print(f"DEBUG: Length of query_embedding_values: {len(query_embedding_values)}")
        if query_embedding_values: # Check if list is not empty before accessing elements
            print(f"DEBUG: Type of first element in list: {type(query_embedding_values[0])}")
            print(f"DEBUG: Sample query_embedding_values (first 5 elements): {query_embedding_values[:min(5, len(query_embedding_values))]}")
    # --- END DEBUGGING PRINTS ---

    conn = get_db_connection() # This will ensure pgvector is registered for 'conn'
    cur = conn.cursor()
    query_sql = """
    SELECT 
        id, description, cluster_id, 
        created_at, updated_at, status, priority, metadata
    FROM 
        incidents
    ORDER BY 
        embedding <-> %s::vector
    LIMIT 5;
    """
    try:
        cur.execute(query_sql, (query_embedding_values,))
        results = cur.fetchall()
        incident_results = []
        for row in results:
            # Safely handle cluster_id conversion
            cluster_id_value = int(row[2]) if isinstance(row[2], (int, np.integer)) else row[2]
            
            incident_results.append({
                "id": row[0],
                "description": row[1],
                "cluster_id": cluster_id_value,
                "created_at": row[3].isoformat() if isinstance(row[3], datetime) else str(row[3]),
                "updated_at": row[4].isoformat() if isinstance(row[4], datetime) else str(row[4]),
                "status": row[5],
                "priority": row[6],
                "metadata": row[7]
            })
        print(f"Found {len(incident_results)} similar incidents via tool.")
        return incident_results
    except Exception as e:
        print(f"Database query failed during tool execution: {e}")
        # Return a more descriptive error if possible from your app
        return [{"error": f"Database query failed: {str(e)}", "details": str(e)}]
    finally:
        cur.close()

# --- MODIFIED: How tools are passed to the model and chat session ---
@app.post("/query")
async def query_incidents_with_llm(request_body: QueryRequest):
    user_query = request_body.query_text
    try:
        # Define the tool (FunctionDeclaration)
        incident_search_tool = Tool(
            function_declarations=[
                FunctionDeclaration(
                    name="fetch_similar_incidents_from_db",
                    description="Fetches incidents from a database based on semantic similarity to a query. Use this tool when the user asks for incidents, reports, or problems that are similar to a given description or keyword.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "query_text": {"type": "string", "description": "The natural language query to use for semantic search in the incident database."},
                        },
                        "required": ["query_text"],
                    },
                )
            ]
        )
        
        # Get the generative model, passing the tools at this stage
        # The tools parameter here registers the tools with the model
        model = get_generative_model(tools_to_register=[incident_search_tool])
        
        # Start the chat *without* the tools argument, as they are already registered with the model
        chat = model.start_chat() 
        
        print(f"Sending user query to LLM: '{user_query}'")
        response = chat.send_message(user_query)
        
        tool_call_part = None
        for part in response.candidates[0].content.parts:
            if part.function_call:
                tool_call_part = part
                break
        if tool_call_part:
            function_call = tool_call_part.function_call
            function_name = function_call.name
            function_args = {k: v for k, v in function_call.args.items()}
            print(f"LLM decided to call tool: {function_name} with args: {function_args}")
            if function_name == "fetch_similar_incidents_from_db":
                tool_output = fetch_similar_incidents_from_db(**function_args)
                print(f"Sending tool output back to LLM: {tool_output}")
                response = chat.send_message(
                    Part.from_function_response(name=function_name, response={"result": tool_output})
                )
                final_answer = response.candidates[0].content.text
            else:
                final_answer = f"Error: LLM requested an unknown tool: {function_name}"
        else:
            final_answer = response.candidates[0].content.text
            print("LLM generated a direct text response (no tool call).")
        print(f"Final LLM Answer: {final_answer}")
        return JSONResponse(content={"answer": final_answer}, status_code=200)

    except RuntimeError as re:
        print(f"Runtime error during Cloud Function execution: {re}")
        raise HTTPException(status_code=500, detail=str(re))
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")

# --- Local Testing Block (only runs when executed directly) ---
if __name__ == "__main__":
    import uvicorn
    print("--- Starting FastAPI app locally ---")
    print(f"DB_HOST: {os.environ.get('DB_HOST')}")
    print(f"DB_NAME: {os.environ.get('DB_NAME')}")
    print(f"EMBEDDING_MODEL_NAME: {os.environ.get('EMBEDDING_MODEL_NAME')}")
    print(f"GEMINI_MODEL_NAME: {os.environ.get('GEMINI_MODEL_NAME')}")
    print("------------------------------------")
    # For local running, use uvicorn.run directly
    uvicorn.run(app, host="0.0.0.0", port=8000)
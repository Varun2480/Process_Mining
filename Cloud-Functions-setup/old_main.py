import os
import json
import psycopg2
from pgvector.psycopg2 import register_vector
from vertexai.language_models import TextEmbeddingModel, TextEmbedding
# Removed Schema and Type from this import
from vertexai.preview.generative_models import GenerativeModel, Tool, Part, GenerationResponse 
# Removed the second import line for Schema and Type completely
from datetime import datetime
import numpy as np

# FastAPI specific imports
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add this line for local testing, remove or comment out for deployment
# from dotenv import load_dotenv
# load_dotenv()

# --- Configuration (from Environment Variables) ---
DB_HOST = os.environ.get('DB_HOST') 
DB_NAME = os.environ.get('DB_NAME')
DB_USER = os.environ.get('DB_USER')
DB_PASSWORD = os.environ.get('DB_PASSWORD')
DB_PORT = os.environ.get('DB_PORT', '5432') 

EMBEDDING_MODEL_NAME = "text-embedding-004"
EMBEDDING_DIMENSION = 768 
GEMINI_MODEL_NAME = "gemini-1.5-flash-001" 

# Global instances for warm starts
embedding_model_instance = None
generative_model_instance = None
_db_conn_pool = None 

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Incident Semantic Search with Gemini 1.5 and PgVector",
    description="API for querying incident descriptions using semantic search powered by Vertex AI Gemini and PgVector.",
    version="1.0.0",
)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models for Request/Response ---
class QueryRequest(BaseModel):
    query_text: str

# --- Helper Functions (mostly from previous code) ---

def get_embedding_model():
    """Initializes and returns the Vertex AI embedding model."""
    global embedding_model_instance
    if embedding_model_instance is None:
        try:
            embedding_model_instance = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL_NAME)
            print(f"Embedding model '{EMBEDDING_MODEL_NAME}' initialized.")
        except Exception as e:
            print(f"Failed to initialize embedding model: {e}")
            raise RuntimeError("Could not initialize embedding model.") from e
    return embedding_model_instance

def get_generative_model():
    """Initializes and returns the Vertex AI GenerativeModel."""
    global generative_model_instance
    if generative_model_instance is None:
        try:
            generative_model_instance = GenerativeModel(GEMINI_MODEL_NAME)
            print(f"Generative model '{GEMINI_MODEL_NAME}' initialized.")
        except Exception as e:
            print(f"Failed to initialize generative model: {e}")
            raise RuntimeError("Could not initialize generative model.") from e
    return generative_model_instance

def get_db_connection():
    """Establishes and returns a connection to the PostgreSQL database.
    Registers pgvector type for the connection."""
    global _db_conn_pool
    
    if _db_conn_pool is None:
        try:
            conn = psycopg2.connect(
                host=DB_HOST,
                database=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD,
                port=DB_PORT
            )
            register_vector(conn)
            print("Database connection established and pgvector type registered.")
            _db_conn_pool = conn
            
        except Exception as e:
            print(f"Failed to connect to database: {e}")
            raise RuntimeError("Could not connect to database.") from e
    
    return _db_conn_pool

def get_gecko_embedding(texts: str | list[str]) -> list[float] | list[list[float]]:
    """
    Generates embeddings for the given text(s) using Vertex AI Gecko model.
    Handles both single string and a list of strings for batch processing.
    """
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

# --- Core Database Search Function (called by LLM via tool) ---
def fetch_similar_incidents_from_db(query_text: str) -> list[dict]:
    """
    Fetches similar incidents from the PostgreSQL database using vector similarity.
    This function is intended to be called by the LLM as a tool.
    """
    print(f"Tool call: Fetching similar incidents for query: '{query_text}'")
    
    query_embedding_values = get_gecko_embedding(query_text)
    
    if not query_embedding_values:
        print("Failed to generate embedding for the query. Returning empty results.")
        return []

    conn = get_db_connection()
    cur = conn.cursor()

    query_sql = """
    SELECT 
        id, description, cluster_id, 
        created_at, updated_at, status, priority, metadata
    FROM 
        incidents
    ORDER BY 
        embedding <-> %s
    LIMIT 5;
    """
    
    try:
        cur.execute(query_sql, (query_embedding_values,))
        results = cur.fetchall()

        incident_results = []
        for row in results:
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
        return [{"error": f"Database query failed: {str(e)}", "details": str(e)}]
    finally:
        cur.close()


# --- FastAPI Endpoint ---
@app.post("/query")
async def query_incidents_with_llm(request_body: QueryRequest):
    """
    Receives a natural language query, uses Gemini 1.5 with function calling
    to retrieve relevant incidents from PgVector, and returns a synthesized answer.
    """
    user_query = request_body.query_text
    
    try:
        model = get_generative_model()

        # --- UPDATED: Simplified Tool Parameter Definition ---
        incident_search_tool = Tool(
            function_declarations=[
                FunctionDeclaration(
                    name="fetch_similar_incidents_from_db",
                    description="Fetches incidents from a database based on semantic similarity to a query. Use this tool when the user asks for incidents, reports, or problems that are similar to a given description or keyword.",
                    parameters={ # Directly use dictionary for parameters
                        "type": "object",
                        "properties": {
                            "query_text": {"type": "string", "description": "The natural language query to use for semantic search in the incident database."},
                        },
                        "required": ["query_text"],
                    },
                )
            ]
        )

        chat = model.start_chat(tools=[incident_search_tool])
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

# This is the entry point for Google Cloud Functions
# The function name MUST match the entry-point specified during deployment
def main(request: Request):
    """
    Cloud Function entry point for FastAPI application.
    """
    return app(request._scope, request._receive, request._send)

# --- Local Testing Block (only runs when executed directly) ---
if __name__ == "__main__":
    import uvicorn
    print("--- Starting FastAPI app locally ---")
    print(f"DB_HOST: {os.environ.get('DB_HOST')}")
    print(f"DB_NAME: {os.environ.get('DB_NAME')}")
    print(f"EMBEDDING_MODEL_NAME: {os.environ.get('EMBEDDING_MODEL_NAME')}")
    print(f"GEMINI_MODEL_NAME: {os.environ.get('GEMINI_MODEL_NAME')}")
    print("------------------------------------")
    uvicorn.run(app, host="0.0.0.0", port=8000)
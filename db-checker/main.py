import os
import psycopg2
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
# from dotenv import load_dotenv # For local testing

# Load environment variables from .env file for LOCAL testing only
# Cloud Run will inject these via --set-env-vars
# load_dotenv()

# --- Configuration (from Environment Variables) ---
# These should be passed correctly by Cloud Run deployment
DB_HOST = os.environ.get('DB_HOST')
DB_NAME = os.environ.get('DB_NAME')
DB_USER = os.environ.get('DB_USER')
DB_PASSWORD = os.environ.get('DB_PASSWORD')
DB_PORT = os.environ.get('DB_PORT', '5432')

app = FastAPI(title="Database Connection Checker")

@app.get("/db_check")
async def db_check():
    conn = None
    cur = None
    try:
        # Print environment variables to Cloud Run logs for debugging
        print(f"Attempting to connect with:")
        print(f"  DB_HOST: {DB_HOST}")
        print(f"  DB_NAME: {DB_NAME}")
        print(f"  DB_USER: {DB_USER}")
        print(f"  DB_PORT: {DB_PORT}")
        # DO NOT print DB_PASSWORD to logs in production!

        conn = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            port=DB_PORT,
            connect_timeout=10 # Increased timeout for robustness
        )
        cur = conn.cursor()
        cur.execute("SELECT version();") # A simple query to check connection
        db_version = cur.fetchone()[0]
        print(f"Successfully connected to database. Version: {db_version}")
        return JSONResponse(content={"status": "success", "db_version": db_version})
    except Exception as e:
        print(f"Failed to connect to database or execute query: {e}")
        raise HTTPException(status_code=500, detail=f"Database connection error: {e}")
    finally:
        # Ensure connection and cursor are closed properly
        if cur:
            cur.close()
        if conn:
            conn.close()

# --- Local Testing Block (only runs when executed directly) ---
if __name__ == "__main__":
    import uvicorn
    print("--- Starting simple DB checker locally ---")
    print(f"DB_HOST: {os.environ.get('DB_HOST')}")
    print(f"DB_NAME: {os.environ.get('DB_NAME')}")
    print(f"DB_USER: {os.environ.get('DB_USER')}")
    print("------------------------------------------")
    uvicorn.run(app, host="0.0.0.0", port=8000)
This guide will walk you through setting up your semantic search application on Google Cloud, from project creation to deployment and testing. We'll ensure all necessary components, including **Cloud SQL (PostgreSQL with `pgvector`)** and **Vertex AI (Gemini 1.5 Flash for LLM and text embeddings)**, are correctly configured.

**Assumptions:**

* You have a Google Cloud account.
* You are comfortable with basic command-line operations (PowerShell/CMD on Windows, Terminal on macOS/Linux).

---

## Step-by-Step Setup Guide

### Section 1: Google Cloud Project Setup

This is where you'll create a new home for your project in Google Cloud.

1.  **Create a New Google Cloud Project:**
    * Go to the Google Cloud Console: [https://console.cloud.google.com/](https://console.cloud.google.com/)
    * Click on the **project selector dropdown** at the top (it usually shows your current project name or "My First Project").
    * Click **"New Project"**.
    * Enter a **"Project name"** (e.g., `IncidentSemanticSearch`). The **Project ID** will be generated automatically (e.g., `incident-semantic-search-123456`). **Note this Project ID down, as you'll use it frequently.**
    * Click **"Create"**.
    * Once the project is created, make sure you **select your new project** from the project selector dropdown.

2.  **Enable Required Google Cloud APIs:**
    These APIs allow your project to use specific Google Cloud services.
    * In the Google Cloud Console, navigate to **"APIs & Services" > "Enabled APIs & Services"**.
    * Click **"+ ENABLE APIS AND SERVICES"**.
    * Search for and enable the following APIs, one by one:
        * `Cloud SQL Admin API`
        * `Vertex AI API`
        * `Cloud Run Admin API`
        * `Cloud Build API`
        * `Artifact Registry API`

---

### Section 2: Cloud SQL PostgreSQL Database Setup

Here, you'll set up your PostgreSQL database, which will store your incident data and embeddings.

1.  **Create a Cloud SQL PostgreSQL Instance:**
    * In the Google Cloud Console, navigate to **"SQL"**.
    * Click **"+ CREATE INSTANCE"**.
    * Choose **"PostgreSQL"**.
    * **Choose a password** for the default `postgres` user. **SAVE THIS PASSWORD SECURELY.**
    * Set an **"Instance ID"** (e.g., `my-incident-db`). This is the name of your database server.
    * Choose a **"Region"** (e.g., `us-central1`). **Make sure this matches the region you will deploy your application to later.**
    * Under "Choose a database version," keep the latest stable version (e.g., `PostgreSQL 14` or `15`).
    * Under "Configuration options" > "Connectivity":
        * Select **"Public IP"**.
        * To allow your local machine to connect for setup, under "Authorized networks," click **"ADD NETWORK"**.
            * For "Network name," type `My Home IP` (or similar).
            * For "Network," enter **your current public IP address** (you can find this by searching "What is my IP" on Google, or using `curl checkip.amazonaws.com` in your terminal). Add `/32` at the end (e.g., `203.0.113.42/32`).
            * Click **"DONE"**.
            * **Note:** For increased security, you should remove this `0.0.0.0/0` access after initial setup or if you're only deploying to Cloud Run (Cloud Run connects internally).
    * Review other settings (e.g., machine type, storage) and adjust if needed, but defaults are often fine for testing.
    * Click **"CREATE INSTANCE"**. This will take a few minutes.

2.  **Note Down Database Connection Details:**
    Once your Cloud SQL instance is created:
    * Click on your **instance ID** (e.g., `my-incident-db`).
    * On the "Overview" page, note the **"Public IP address"** (e.g., `34.66.49.57`). This is your `DB_HOST`.
    * The default database name is often `postgres`, but we'll create `incidents_db`.
    * Your user is `postgres` and you set the password earlier.
    * The default port is `5432`.

3.  **Create the Database and Enable `pgvector` Extension:**
    You'll need a tool to connect to PostgreSQL. The easiest is `psql` if you have it installed locally, or you can use the Cloud Shell.

    * **Using Cloud Shell (easiest way from GCP):**
        * Click the **"Activate Cloud Shell"** icon in the top right corner of the Google Cloud Console (it looks like `>_`).
        * Once Cloud Shell initializes, connect to your database instance:
            ```bash
            gcloud sql connect my-incident-db --user=postgres
            ```
            Replace `my-incident-db` with your instance ID.
            Enter the `postgres` user password when prompted.
        * Once connected to the `postgres` prompt (`postgres=>`), run these SQL commands:
            ```sql
            CREATE DATABASE incidents_db;
            \c incidents_db;
            CREATE EXTENSION vector;
            ```
            This creates your `incidents_db` database and enables the `pgvector` extension for similarity search.
        * Type `\q` and press Enter to exit `psql`.

---

### Section 3: Local Development Environment Setup

This section prepares your computer to run and test the application code.

1.  **Install Python:**
    * Ensure you have **Python 3.11** installed. Download it from [python.org](https://www.python.org/downloads/). During installation, select "Add Python to PATH."

2.  **Create Your Project Folder and Files:**
    * Create a new folder on your computer for your project (e.g., `Cloud-Functions-setup`).
    * Inside this folder, create the following empty files:
        * `main.py`
        * `requirements.txt`
        * `Procfile`
        * `.env`

3.  **Populate `requirements.txt`:**
    This file lists all the Python libraries your application needs.
    * Open `requirements.txt` and add these lines:
        ```
        fastapi
        uvicorn[standard]
        psycopg2-binary
        pgvector
        google-cloud-aiplatform[generative]==1.52.0
        python-dotenv
        ```

4.  **Populate `Procfile`:**
    This file tells Cloud Run how to start your FastAPI application.
    * Open `Procfile` and add this single line:
        ```
        web: uvicorn main:app --host 0.0.0.0 --port $PORT
        ```

5.  **Populate `.env` (for Local Testing Only):**
    This file holds your database connection details for local running, so you don't hardcode them in your `main.py`. **This file should NOT be committed to public repositories.**
    * Open `.env` and add your database details. Use `localhost` for `DB_HOST` because you'll connect via the Cloud SQL Proxy.
        ```
        DB_HOST="localhost"
        DB_NAME="incidents_db"
        DB_USER="postgres"
        DB_PASSWORD="YOUR_POSTGRES_PASSWORD" # Use the password you set in Cloud SQL
        DB_PORT="5432"

        EMBEDDING_MODEL_NAME="text-embedding-004"
        EMBEDDING_DIMENSION="768"
        GEMINI_MODEL_NAME="gemini-1.5-flash-001"
        ```

6.  **Populate `main.py` (Your Application Code):**
    This is the core Python code for your FastAPI application, including LLM interaction and database logic.
    * Open `main.py` and copy-paste the entire code block below:

    ```python
    # main.py

    import os
    import json
    import psycopg2
    from pgvector.psycopg2 import register_vector
    from vertexai.language_models import TextEmbeddingModel, TextEmbedding
    from vertexai.preview.generative_models import GenerativeModel, Tool, Part, FunctionDeclaration, GenerationResponse 
    from datetime import datetime 
    import numpy as np 

    # FastAPI specific imports
    from fastapi import FastAPI, Request, HTTPException 
    from fastapi.responses import JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel

    # IMPORTANT: Load environment variables from .env file for LOCAL testing
    from dotenv import load_dotenv
    load_dotenv() 

    # --- Configuration (from Environment Variables) ---
    DB_HOST = os.environ.get('DB_HOST') 
    DB_NAME = os.environ.get('DB_NAME')
    DB_USER = os.environ.get('DB_USER')
    DB_PASSWORD = os.environ.get('DB_PASSWORD')
    DB_PORT = os.environ.get('DB_PORT', '5432') 

    EMBEDDING_MODEL_NAME = os.environ.get('EMBEDDING_MODEL_NAME', "text-embedding-004")
    EMBEDDING_DIMENSION = int(os.environ.get('EMBEDDING_DIMENSION', "768"))
    GEMINI_MODEL_NAME = os.environ.get('GEMINI_MODEL_NAME', "gemini-1.5-flash-001")

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

    def get_generative_model(tools_to_register=None):
        global generative_model_instance
        if generative_model_instance is None:
            try:
                generative_model_instance = GenerativeModel(GEMINI_MODEL_NAME, tools=tools_to_register)
                print(f"Generative model '{GEMINI_MODEL_NAME}' initialized with tools.")
            except Exception as e:
                print(f"Failed to initialize generative model: {e}")
                raise RuntimeError("Could not initialize generative model.") from e
        return generative_model_instance

    def get_db_connection():
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

    # --- FastAPI Endpoint (@app.post("/query")) ---
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
        uvicorn.run(app, host="0.0.0.0", port=8000)
    ```

7.  **Set Up Python Virtual Environment and Install Dependencies:**
    It's good practice to use a virtual environment to manage project dependencies.
    * Open your **terminal** (PowerShell/CMD on Windows, Terminal on macOS/Linux).
    * **Navigate to your project folder** (e.g., `cd G:\POC's\Process Mining\Process_Mining\Cloud-Functions-setup`).
    * Create a virtual environment:
        ```bash
        python -m venv venv_process_mining
        ```
    * Activate the virtual environment:
        * **Windows:** `.\venv_process_mining\Scripts\activate`
        * **macOS/Linux:** `source venv_process_mining/bin/activate`
        (You'll see `(venv_process_mining)` appear at the start of your terminal prompt.)
    * Install the dependencies listed in `requirements.txt`:
        ```bash
        pip install -r requirements.txt
        ```

---

### Section 4: Local Testing with Cloud SQL Proxy (Optional but Recommended)

This allows you to run your application on your computer and connect securely to your Cloud SQL database.

1.  **Download Cloud SQL Proxy:**
    * Go to: [https://cloud.google.com/sql/docs/postgres/connect-external-app#connect-proxy](https://cloud.google.com/sql/docs/postgres/connect-external-app#connect-proxy)
    * Download the appropriate executable for your OS (e.g., `cloud_sql_proxy.exe` for Windows).
    * Place the downloaded executable in your **project folder** (where `main.py` is).

2.  **Run the Cloud SQL Proxy:**
    * Open a **new terminal window** (keep your existing one for your Python app).
    * **Navigate to your project folder.**
    * Run the proxy command. Replace `YOUR_PROJECT_ID`, `YOUR_REGION`, and `YOUR_INSTANCE_NAME` with your actual details:
        ```bash
        # For Windows (PowerShell/CMD)
        .\cloud_sql_proxy.exe -instances="YOUR_PROJECT_ID:YOUR_REGION:YOUR_INSTANCE_NAME"=tcp:5432
        
        # Example using your project details:
        .\cloud_sql_proxy.exe -instances="poetic-analog-460707-i9:us-central1:my-incident-db"=tcp:5432
        ```
        The proxy will start listening on `localhost:5432`. **Do not close this terminal.**

3.  **Run Your FastAPI Application Locally:**
    * In your **first terminal window** (where your virtual environment is activated), run:
        ```bash
        python main.py
        ```
        You should see messages from Uvicorn indicating the app is running on `http://127.0.0.1:8000`.

4.  **Test Locally with `curl`:**
    * Open a **third terminal window** (or use a tool like Postman/Insomnia).
    * Send a POST request to your local app:
        ```bash
        # For Windows (PowerShell/CMD)
        curl -X POST "[http://127.0.0.1:8000/query](http://127.0.0.1:8000/query)" ^
          -H "Content-Type: application/json" ^
          -d "{\"query_text\": \"Tell me about incidents related to storage capacity alert.\"}"
        ```
    * You should see a JSON response. Check the logs in your `python main.py` terminal for output from your `print` statements.

---

### Section 5: Deploy to Google Cloud Run

This is where you'll make your application available on Google Cloud.

1.  **Install and Configure `gcloud` CLI:**
    This command-line tool allows you to interact with Google Cloud services.
    * Follow the official installation guide: [https://cloud.google.com/sdk/docs/install](https://cloud.google.com/sdk/docs/install)
    * Once installed, open a new terminal and initialize it:
        ```bash
        gcloud init
        ```
        Follow the prompts to log in with your Google account and select your project (`poetic-analog-460707-i9`).

2.  **Grant Cloud Run Service Account Permissions (CRITICAL!):**
    Your deployed application needs permission to talk to Vertex AI and Cloud SQL. Cloud Run services run under a special Google-managed service account.
    * Go to the Google Cloud Console: [https://console.cloud.google.com/](https://console.cloud.google.com/)
    * Navigate to **"IAM & Admin" > "IAM"**.
    * Look for a service account named something like `PROJECT_NUMBER-compute@developer.gserviceaccount.com` (e.g., `1065449696673-compute@developer.gserviceaccount.com`). This is your default Compute Engine service account, often used by Cloud Run.
    * Click the **pencil icon (Edit principal)** next to this service account.
    * Click **"+ ADD ANOTHER ROLE"** and add the following roles:
        * `Cloud SQL Client`
        * `Vertex AI User`
    * Click **"SAVE"**.

3.  **Deploy Your Application to Cloud Run:**
    * Open your terminal and **navigate to your project folder** (where `main.py`, `requirements.txt`, and `Procfile` are).
    * **Ensure your Python virtual environment is DEACTIVATED** for deployment (`deactivate` command). This ensures `gcloud` uses your system Python and doesn't get confused by the virtual env.
    * Run the deployment command. Replace `YOUR_PROJECT_ID` with your actual Project ID.
        ```bash
        gcloud run deploy incident-query-endpoint \
          --source=. \
          --region us-central1 \
          --memory 1Gi \
          --cpu 1 \
          --timeout 300s \
          --allow-unauthenticated \
          --set-env-vars DB_HOST="34.66.49.57",DB_NAME="incidents_db",DB_USER="postgres",DB_PASSWORD="postgres" \
          --project="poetic-analog-460707-i9"
        ```
        * `incident-query-endpoint`: This will be the name of your Cloud Run service.
        * `--source=.`: Tells Cloud Run to build your container from the current directory. It will automatically detect your `Procfile`.
        * `--region us-central1`: Make sure this matches your Cloud SQL region.
        * `--memory 1Gi --cpu 1`: Sets resources for your service.
        * `--timeout 300s`: Sets the request timeout.
        * `--allow-unauthenticated`: Makes your service publicly accessible (for testing). For production, you'd likely remove this.
        * `--set-env-vars`: Passes your database credentials directly to the Cloud Run environment. **Crucially, the `DB_HOST` here is the PUBLIC IP of your Cloud SQL instance, not `localhost` like in your `.env` file.**
        * `--project`: Specifies your Google Cloud Project ID.

    * The command will take a few minutes to build and deploy. Once complete, it will provide a **Service URL** (e.g., `https://incident-query-endpoint-1065449696673.us-central1.run.app`). **Note this URL down.**

---

### Section 6: Test the Deployed Cloud Run Service

Now, let's verify that your application is working correctly in the cloud.

1.  **Test with `curl`:**
    * Open your terminal (any terminal, no need for virtual env or proxy).
    * Use the `Service URL` provided by the `gcloud run deploy` command and append `/query` to it.
        ```bash
        # For Windows (PowerShell/CMD)
        curl -X POST "[https://incident-query-endpoint-1065449696673.us-central1.run.app/query](https://incident-query-endpoint-1065449696673.us-central1.run.app/query)" ^
          -H "Content-Type: application/json" ^
          -d "{\"query_text\": \"Tell me about incidents related to storage capacity alert.\"}"
        ```
    * You should receive a JSON response from your deployed FastAPI application.

2.  **Check Cloud Run Logs:**
    * In the Google Cloud Console, navigate to **"Cloud Run"**.
    * Select your `incident-query-endpoint` service.
    * Click on the **"Logs"** tab.
    * You should see detailed logs from your application, including messages about the generative model being initialized, the LLM interacting, and database queries being performed. This is your main debugging tool if you encounter any issues.
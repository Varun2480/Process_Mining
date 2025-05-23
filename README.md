
### Step-by-Step Approach: AI-Powered Incident Analysis with `pgvector` and Vertex AI

**Goal:** To embed 100 incident descriptions, cluster them using K-means, store them in `pgvector`, and then query this store using Vertex AI for similarity search.

**Key Components:**
* **Vertex AI Embeddings API (Gecko):** For generating text embeddings.
* **Google Cloud SQL for PostgreSQL + `pgvector` extension:** For scalable vector storage.
* **Vertex AI (e.g., Cloud Functions, Cloud Run):** For orchestrating the query flow and serving as the API endpoint.
* **Google Cloud Storage (Optional but recommended):** For storing raw data or larger documents linked to incidents.

---

#### Phase 1: Data Preparation & GCP Project Setup

1.  **Google Cloud Project Setup:**
    * Create a new Google Cloud Project or select an existing one.
    * Enable necessary APIs:
        * **Vertex AI API**
        * **Cloud SQL Admin API**
        * **Cloud Functions API** (if using Cloud Functions)
        * **Cloud Run API** (if using Cloud Run)
        * **Cloud Storage API** (if using GCS)
    * Set up billing for your project.

2.  **Prepare Incident Data:**
    * Organize your 100 incident descriptions (and any associated metadata like `id`, `created_at`, `updated_at`, etc.) into a structured format (e.g., CSV, JSON).
    * Ensure each incident has a unique identifier.

#### Phase 2: `pgvector` Database Setup on Google Cloud SQL

1.  **Create a Cloud SQL for PostgreSQL Instance:**
    * Go to **Cloud SQL** in your GCP Console.
    * Click "Create instance" and choose "PostgreSQL."
    * Select a suitable database version (e.g., PostgreSQL 14 or higher).
    * Configure instance ID, region, machine type, storage, and set a strong root password.
    * **Crucially, configure networking:** Decide if you'll use Public IP (for quick setup, requires authorized networks) or Private IP (more secure, requires VPC Access Connector). For production, Private IP is highly recommended.
2.  **Connect to Your PostgreSQL Instance:**
    * You can use `gcloud sql connect` from Cloud Shell, `psql` locally, or a database client (DBeaver, DataGrip) after configuring authorized networks if using Public IP.
3.  **Install the `pgvector` Extension:**
    * Once connected to your Cloud SQL instance, execute the following SQL command:
        ```sql
        CREATE EXTENSION vector;
        ```
    * This enables vector capabilities within your database.
4.  **Create Your Incidents Table with Vector Column:**
    * Define a table that will store your incident data, including a column specifically for the embeddings.
    * **Determine Embedding Dimension:** The Gecko models (like `text-embedding-004`) typically produce embeddings of a specific dimension (e.g., 768 or 1536). You'll need to know this value.
    * **Example SQL Schema (Conceptual):**
        ```sql
        CREATE TABLE incidents (
            id VARCHAR(255) PRIMARY KEY,
            description TEXT NOT NULL,
            embedding VECTOR(<YOUR_EMBEDDING_DIMENSION>), -- e.g., VECTOR(768)
            cluster_id INT, -- To store K-means cluster assignment
            created_at TIMESTAMP WITH TIME ZONE,
            updated_at TIMESTAMP WITH TIME ZONE,
            -- Add other relevant metadata columns
            metadata JSONB -- For flexible storage of additional incident details
        );
        ```
        *Replace `<YOUR_EMBEDDING_DIMENSION>` with the actual dimension of your embeddings.*
5.  **Add Indexes (Crucial for Performance):**
    * **Concept:** For efficient similarity search on large datasets, you need to add an index to your `embedding` column. `pgvector` supports various index types like `IVFFlat` or `HNSW`.
    * **Process:** Choose an appropriate index based on your needs (e.g., `CREATE INDEX ON incidents USING ivfflat (embedding vector_l2_ops) WITH (lists = 100);`).
    * **Reference:** Refer to the `pgvector` documentation for index creation best practices.

#### Phase 3: Embedding Generation (Gecko) & Data Loading

1.  **Authentication to Vertex AI:**
    * Ensure your environment (where you run the embedding script) is authenticated to GCP. This usually involves:
        * For local development: `gcloud auth application-default login`
        * For GCP services (Cloud Functions, Cloud Run): Using default service account credentials.
2.  **Write an Embedding and Data Loading Script:**
    * Use a programming language (Python is common) with the Google Cloud Client Library for Vertex AI and a PostgreSQL client library (e.g., `psycopg2`).
    * **Conceptual Python Flow:**
        ```python
        # 1. Initialize Vertex AI Embedding Model:
        #    from vertexai.language_models import TextEmbeddingModel
        #    model = TextEmbeddingModel.from_pretrained("text-embedding-004")

        # 2. Connect to PostgreSQL (Cloud SQL instance):
        #    import psycopg2
        #    conn = psycopg2.connect(
        #        host="YOUR_CLOUD_SQL_IP_OR_CONNECTION_STRING",
        #        database="your_database_name",
        #        user="postgres", # or your custom user
        #        password="your_db_password",
        #        port="5432"
        #    )
        #    cur = conn.cursor()
        #    # Important: Register vector type for psycopg2 to handle array data
        #    from psycopg2.extras import register_vector
        #    register_vector(psycopg2.extensions.VECTORS)

        # 3. Loop through your 100 incidents:
        #    For each incident_data_row in your 100 incidents:
        #        a. Get incident description: `description = incident_data_row['description']`
        #        b. Generate embedding using Vertex AI:
        #           `embedding = model.get_embeddings([description])[0].values`
        #        c. (Optional) Perform K-means clustering:
        #           If doing K-means on the fly or after collecting all embeddings:
        #           `from sklearn.cluster import KMeans`
        #           `kmeans = KMeans(n_clusters=K_VALUE, random_state=0, n_init='auto').fit(all_embeddings)`
        #           `cluster_id = kmeans.labels_[current_incident_index]`
        #        d. Prepare data for insertion:
        #           `incident_id = incident_data_row['id']`
        #           `created_at = incident_data_row['created_at']` (convert to datetime object)
        #           `updated_at = incident_data_row['updated_at']` (convert to datetime object)
        #           `metadata_json = json.dumps(incident_data_row['other_metadata'])`
        #        e. Execute INSERT query into `pgvector`:
        #           `insert_query = """`
        #           `INSERT INTO incidents (id, description, embedding, cluster_id, created_at, updated_at, metadata)`
        #           `VALUES (%s, %s, %s, %s, %s, %s, %s);`
        #           `"""`
        #           `cur.execute(insert_query, (incident_id, description, embedding, cluster_id, created_at, updated_at, metadata_json))`
        #
        # 4. Commit and Close:
        #    `conn.commit()`
        #    `cur.close()`
        #    `conn.close()`
        ```
    * **Run the Script:** Execute this script from your local machine or a GCP environment (e.g., a Vertex AI Workbench notebook, a Cloud Build job) to populate your `pgvector` table.

#### Phase 4: Querying with Vertex AI (API Endpoint)

This phase involves creating a service that can receive a user's query, embed it, query `pgvector`, and return results.

1.  **Choose a Deployment Method on Vertex AI:**
    * **Google Cloud Functions (Recommended for simplicity):**
        * Good for stateless, event-driven functions.
        * You'll create an HTTP-triggered function that receives user queries.
    * **Google Cloud Run (Recommended for more complex APIs/Scalability):**
        * For containerized applications, more control over environment.
        * You'll deploy a Docker image that serves an HTTP endpoint.
2.  **Develop the Query Service Logic:**
    * **Authentication:** The service account for your Cloud Function/Run instance needs specific IAM roles:
        * `Vertex AI User` role (for calling the embedding model).
        * `Cloud SQL Client` role (for connecting to Cloud SQL).
    * **Conceptual Python Code for an HTTP Handler (e.g., `main.py`):**
        ```python
        # Import necessary libraries (vertexai, psycopg2, json, os)
        # from vertexai.language_models import TextEmbeddingModel
        # import psycopg2
        # from psycopg2.extras import register_vector
        # import json
        # import os

        # Initialize embedding model globally (outside function for efficiency)
        # embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")
        # register_vector(psycopg2.extensions.VECTORS) # Only needs to be done once per process

        # Define database connection details using environment variables for security
        # DB_HOST = os.environ.get('DB_HOST') # For Cloud Functions/Run, typically project:region:instance_name for Unix socket
        # DB_NAME = os.environ.get('DB_NAME')
        # DB_USER = os.environ.get('DB_USER')
        # DB_PASSWORD = os.environ.get('DB_PASSWORD')
        # DB_PORT = os.environ.get('DB_PORT', '5432') # Default PostgreSQL port

        # Function to establish DB connection (using Cloud SQL Proxy or Unix socket if possible)
        # def get_db_connection():
        #     return psycopg2.connect(
        #         host=DB_HOST,
        #         database=DB_NAME,
        #         user=DB_USER,
        #         password=DB_PASSWORD,
        #         port=DB_PORT # Or remove if using Unix socket
        #     )

        # Main HTTP handler function (e.g., for Flask or Cloud Functions)
        # def handle_query_request(request):
        #     request_json = request.get_json(silent=True)
        #     query_text = request_json.get('query_text', '')

        #     if not query_text:
        #         return json.dumps({"error": "No query_text provided"}), 400, {'Content-Type': 'application/json'}

        #     try:
        #         # 1. Embed the user query using Gecko
        #         query_embedding = embedding_model.get_embeddings([query_text])[0].values

        #         # 2. Connect to pgvector
        #         conn = get_db_connection()
        #         cur = conn.cursor()

        #         # 3. Perform similarity search (e.g., using L2 distance <->)
        #         #    Order by distance (smaller is more similar) and LIMIT results
        #         query_sql = """
        #         SELECT id, description, cluster_id, created_at, updated_at
        #         FROM incidents
        #         ORDER BY embedding <-> %s
        #         LIMIT 5;
        #         """
        #         cur.execute(query_sql, (query_embedding,))
        #         results = cur.fetchall()

        #         # 4. Format and return results
        #         incident_results = []
        #         for row in results:
        #             incident_results.append({
        #                 "id": row[0],
        #                 "description": row[1],
        #                 "cluster_id": row[2],
        #                 "created_at": str(row[3]), # Convert datetime to string
        #                 "updated_at": str(row[4])
        #                 # Add other fields as needed
        #             })

        #         cur.close()
        #         conn.close()

        #         return json.dumps(incident_results), 200, {'Content-Type': 'application/json'}

        #     except Exception as e:
        #         # Log the error for debugging
        #         print(f"Error processing request: {e}")
        #         return json.dumps({"error": "Internal server error"}), 500, {'Content-Type': 'application/json'}
        ```
3.  **Deploy the Service:**
    * **For Cloud Functions:** Use `gcloud functions deploy` command. Ensure to set environment variables for database connection details (DB_HOST, DB_NAME, DB_USER, DB_PASSWORD) and configure VPC Access Connector if using Private IP for Cloud SQL.
    * **For Cloud Run:** Build a Docker image of your application and deploy it using `gcloud run deploy`. Configure environment variables and ensure the service account has the necessary permissions.

#### Phase 5: Testing and Integration

1.  **Test the Endpoint:**
    * Once your Cloud Function or Cloud Run service is deployed, you'll receive an HTTPS endpoint URL.
    * Use `curl`, Postman, or a custom script to send POST requests to this URL with your `query_text` in a JSON payload.
2.  **Integrate with a Frontend (Optional):**
    * Develop a simple web UI (e.g., using React, Vue, or plain HTML/JS) that allows users to input their query text and display the semantic search results retrieved from your Vertex AI endpoint.


This comprehensive plan covers all the major components from data preparation to querying, providing a clear roadmap for setting up your AI-powered incident analysis system. Always refer to Google Cloud's official documentation for specific service configurations, authentication best practices, and security measures.
from google import genai
from google.genai.types import Part
from dotenv import load_dotenv
import os

load_dotenv()

# --- IMPORTANT CONFIGURATION FOR VERTEX AI ---
# When using gs:// URIs, you MUST configure the genai client for Vertex AI.
# This enables direct understanding of gs:// paths by the model.
# Ensure your environment is authenticated to Google Cloud (e.g., via `gcloud auth application-default login`)
# and that the Vertex AI API is enabled in your project.

# Replace with your actual Google Cloud Project ID and Location
# These should match where your Vertex AI resources are located.
YOUR_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "poetic-analog-460707-i9") # Using environment variable for project ID
YOUR_LOCATION = os.getenv("GCP_LOCATION", "us-central1")               # Using environment variable for location

# Configure the genai client to use Vertex AI
# This is the key difference from your original code.
try:
    client = genai.Client(
        project=YOUR_PROJECT_ID,
        location=YOUR_LOCATION,
        # api_version="v1" is implicitly handled by Vertex AI client
    )
    print(f"Configured genai client for Vertex AI: Project '{YOUR_PROJECT_ID}', Location '{YOUR_LOCATION}'")
except Exception as e:
    print(f"Error configuring genai client for Vertex AI: {e}")
    print("Please ensure your Google Cloud environment is authenticated and the specified PROJECT_ID/LOCATION are correct.")
    # Exit or handle the error appropriately
    exit()


# Define the model to use. 'gemini-2.5-flash-preview-05-20' is a valid model for Vertex AI.
MODEL_NAME = "gemini-1.5-flash" # Changed to a generally available and well-supported model
                                 # You can switch back to "gemini-2.5-flash-preview-05-20"
                                 # if it's available in your region and project.
                                 # For general use, gemini-1.5-flash is often a good default.

# The GCS URI of your video
VIDEO_GCS_URI = "gs://process-mining-bucket-1/How to Buy On Amazon (really easy).mp4"
VIDEO_MIME_TYPE = "video/mp4" # Ensure this matches your video file type

# The prompt for the model
PROMPT = "What is in the video? Summarize with crisp and clear words in 5 pointers."

import pdb; pdb.set_trace()
try:
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[
            Part.from_uri(
                file_uri=VIDEO_GCS_URI,
                mime_type=VIDEO_MIME_TYPE,
            ),
            PROMPT, # Prompt is now passed as a string directly, which Part.from_text() handles
        ],
    )
    print("\n--- Video Summary ---")
    print(response.text)

except Exception as e:
    print(f"\nAn error occurred during content generation: {e}")
    print("Please check:")
    print(f"- If the model '{MODEL_NAME}' is available in '{YOUR_LOCATION}' for project '{YOUR_PROJECT_ID}'.")
    print("- Your Google Cloud authentication and permissions to access GCS bucket.")
    print(f"- The GCS URI '{VIDEO_GCS_URI}' is correct and the video file exists.")
    print("- The 'mime_type' for the video is accurate.")
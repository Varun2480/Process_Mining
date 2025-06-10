import functions_framework
import vertexai
from vertexai.generative_models import GenerativeModel, Part
import os
import json

# Global variables for model initialization to optimize cold starts
# These are initialized once per function instance
model = None
PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
GCP_LOCATION = os.environ.get("GCP_LOCATION")

# Define the prompt directly in the code
# You can make this a global constant if you prefer
STAGE_GENERATION_PROMPT = """
Role: You are an expert Process Analyst and Workflow Designer. Your task is to meticulously observe and deconstruct processes from raw input, transforming complex activities into clear, actionable, and sequential stages and steps.

Objective: Analyze the provided video content to identify the complete "as-is" workflow. Your output must clearly delineate the main stages of the process, and for each stage, enumerate the distinct, chronological steps performed within it.

Output Format: Provide the analysis in a structured JSON format, ensuring it is syntactically correct and easily parseable. Always provide theoutput in a JSON format.
example:

{
  "process_title": "A concise, descriptive title for the overall process observed in the video.",
  "video_analysis_summary": "A brief overview (1-2 sentences) of the entire process shown in the video.",
  "stages": [
    {
      "stage_number": 1,
      "stage_name": "Clear, action-oriented name for Stage 1 (e.g., 'Initial Setup', 'Data Input', 'Execution Phase').",
      "stage_description": "A brief description of the primary activity or goal of this stage.",
      "steps": [
        "Step 1.1: Detailed and precise description of the first action.",
        "Step 1.2: Detailed and precise description of the second action.",
        "...",
        "Step 1.N: Last action in this stage."
      ]
    },
    {
      "stage_number": 2,
      "stage_name": "Clear, action-oriented name for Stage 2.",
      "stage_description": "A brief description of the primary activity or goal of this stage.",
      "steps": [
        "Step 2.1: Detailed and precise description of the first action in Stage 2.",
        "...",
        "Step 2.M: Last action in this stage."
      ]
    }
    // ... Continue adding stages until the entire video's process is covered.
  ],
  "completeness_check": "Confirm that all visible actions and the entire duration of the video's process have been analyzed and broken down into stages and steps. (e.g., 'The analysis covers the entire video content provided.')"
}
"""

def _initialize_vertexai_model():
    """Initializes Vertex AI and the GenerativeModel.
    This function should be called once per instance to reduce cold start latency.
    """
    global model
    if model is None:
        if not PROJECT_ID or not GCP_LOCATION:
            raise ValueError(
                "GCP_PROJECT_ID and GCP_LOCATION environment variables must be set."
            )
        vertexai.init(project=PROJECT_ID, location=GCP_LOCATION)
        # Using 1.5 Flash for multimodal video input
        model = GenerativeModel("gemini-2.0-flash-001")
        print(f"Vertex AI initialized for project {PROJECT_ID} in {GCP_LOCATION}")

@functions_framework.http
def process_video_for_workflow(request):
    """HTTP Cloud Function that processes a video GCS URI to extract workflow stages
    and steps using Gemini 1.5 Flash.

    Args:
        request (flask.Request): The request object. Expects a JSON payload
                                 with 'video_gcs_uri'.
    Returns:
        The extracted workflow as JSON, or an error message.
    """
    _initialize_vertexai_model() # Ensure model is initialized

    request_json = request.get_json(silent=True)
    video_gcs_uri = None

    if request_json and 'video_gcs_uri' in request_json:
        video_gcs_uri = request_json['video_gcs_uri']
    else:
        # Also check query parameters for flexibility, though JSON body is preferred
        video_gcs_uri = request.args.get('video_gcs_uri')

    if not video_gcs_uri:
        return 'Missing "video_gcs_uri" parameter in request body (JSON) or query parameters.', 400

    # Create a Part object for the video input
    try:
        video_part = Part.from_uri(mime_type="video/mp4", uri=video_gcs_uri)
    except Exception as e:
        print(f"Error creating video Part from URI: {e}")
        return f"Invalid video GCS URI or format: {e}", 400

    # Combine the video and the embedded prompt
    contents = [
        video_part,
        Part.from_text(STAGE_GENERATION_PROMPT)
    ]

    print(f"Sending request to Gemini 1.5 Flash for video: {video_gcs_uri}")

    # Generate content with the model
    try:
        response = model.generate_content(contents)
        return response.text, 200
            
    except Exception as e:
        print(f"An error occurred during content generation: {e}")
        return f"Error during video processing: {e}", 500
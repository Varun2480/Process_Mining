import vertexai
from vertexai.generative_models import GenerativeModel, Part

def summarize_video_with_gemini_flash(project_id: str, location: str, video_gcs_uri: str, prompt: str) -> str:
    """
    Summarizes a video using Google's Gemini 1.5 Flash model on Vertex AI.

    Args:
        project_id (str): Your Google Cloud project ID.
        location (str): The Google Cloud region where you want to run the model
                        (e.g., "us-central1").
        video_gcs_uri (str): The Google Cloud Storage URI of your video file
                             (e.g., "gs://your-bucket/your-video.mp4").
        prompt (str): The prompt to send to the model for summarization.

    Returns:
        str: The summarized text from the Gemini 1.5 Flash model.
    """

    vertexai.init(project=project_id, location=location)

    # Initialize the Gemini 1.5 Flash model
    # Use 'gemini-1.5-flash-001' or 'gemini-1.5-flash-latest'
    # 'gemini-1.5-flash-001' is the stable version
    # 'gemini-1.5-flash-latest' will always point to the most recent stable version
    # model = GenerativeModel("gemini-1.5-flash-001")
    model = GenerativeModel("gemini-2.0-flash-001")

    # Create a Part object for the video input
    video_part = Part.from_uri(mime_type="video/mp4", uri=video_gcs_uri)

    # Combine the video and text prompt
    contents = [
        video_part,
        Part.from_text(prompt)
    ]

    print(f"Sending request to Gemini 1.5 Flash for video: {video_gcs_uri}")

    # Generate content with the model
    try:
        response = model.generate_content(contents)
        return response.text
    except Exception as e:
        print(f"An error occurred: {e}")
        return f"Error during video summarization: {e}"

if __name__ == "__main__":
    # --- Configuration ---
    YOUR_PROJECT_ID = "poetic-analog-460707-i9"  # Replace with your actual GCP Project ID
    YOUR_GCP_LOCATION = "us-central1"       # Replace with your desired GCP region
    YOUR_VIDEO_GCS_URI = "gs://process-mining-bucket-1/How to Buy On Amazon (really easy).mp4" # Replace with YOUR video's GCS URI

    # Example prompt for video summarization
    # VIDEO_SUMMARIZATION_PROMPT = """
    # Summarize the key events and main topics discussed in the provided video.
    # Focus on important details and provide a concise overview.
    # """

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

    print(f"Attempting to summarize video: {YOUR_VIDEO_GCS_URI}")

    summary = summarize_video_with_gemini_flash(
        project_id=YOUR_PROJECT_ID,
        location=YOUR_GCP_LOCATION,
        video_gcs_uri=YOUR_VIDEO_GCS_URI,
        prompt=STAGE_GENERATION_PROMPT
    )

    print("\n--- Video Summary ---")
    print(summary)
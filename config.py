import os
from dotenv import load_dotenv

# Load environment variables from .env file
def load_config():
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("⚠️ Please set your Google API Key in the .env file as GOOGLE_API_KEY")
    return api_key

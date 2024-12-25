import os
from dotenv import load_dotenv

load_dotenv()

def verify_groq_key():
    """Verify the GROQ API key."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY is missing in the environment variables.")
    return "The token is validated and working good!"

if __name__ == "__main__":
    try:
        # print(f"Loaded API Key: {os.getenv('GROQ_API_KEY')}")  # Debugging line
        message = verify_groq_key()
        print(message)
    except Exception as e:
        print(f"Error: {e}")

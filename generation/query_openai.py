import os
from openai import OpenAI
from dotenv import load_dotenv

# Load variables from .env
load_dotenv()

# Get the API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not set in .env or environment")

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

def query_openai(prompt: str, model: str = "gpt-3.5-turbo") -> str:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=1024,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[OpenAI Error] {str(e)}"

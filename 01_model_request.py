"""Send one message to the configured model. Run beside your local .env file."""

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# Read configuration from the .env beside this script.
load_dotenv(Path(__file__).with_name(".env"), override=True)

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key or api_key == "your_key_here":
    raise SystemExit("Add your OpenRouter API key to the .env beside this script.")

model = os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-120b:free")
base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

# Configure the connection. This line does not send a model request.
client = OpenAI(
    api_key=api_key,
    base_url=base_url,
    timeout=60.0,
    max_retries=0,
)

print(f"Sending one request to: {model}")

# This is where the message is actually sent.
response = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "user", "content": "Reply with exactly: API connection working."}
    ],
    max_tokens=1024,
)

# Extract the first returned answer and display it in the terminal.
print("Reply:", response.choices[0].message.content)
print("Finish reason:", response.choices[0].finish_reason)
if response.usage:
    print("Total tokens:", response.usage.total_tokens)

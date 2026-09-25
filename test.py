from google import genai
from config import GOOGLE_API_KEY

client = genai.Client(api_key=GOOGLE_API_KEY)

print("Available embedding models for your key:")
for model in client.models.list():
    if "embed" in model.name:
        print(f" - {model.name}")
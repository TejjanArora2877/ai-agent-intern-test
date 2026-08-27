import os
import httpx
from pathlib import Path

env_file = Path(".env")
if env_file.exists():
    for line in env_file.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ[k.strip()] = v.strip().strip("\"'")

url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:generateContent"
headers = {"Content-Type": "application/json", "x-goog-api-key": os.getenv("GEMINI_API_KEY")}
res = httpx.post(url, headers=headers, json={"contents": [{"parts": [{"text": "Hello"}]}]})
print("Status:", res.status_code)
try:
    print("Response JSON:", res.json())
except Exception as e:
    print("Response text:", res.text)

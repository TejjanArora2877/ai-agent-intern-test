import os
import httpx
from pathlib import Path

env_file = Path(".env")
if env_file.exists():
    for line in env_file.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ[k.strip()] = v.strip().strip("\"'")

key = os.getenv("GEMINI_API_KEY")
for model in ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-flash-8b", "gemini-3.6-flash"]:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    try:
        res = httpx.post(url, headers={"Content-Type": "application/json", "x-goog-api-key": key}, json={"contents": [{"parts": [{"text": "hi"}]}]})
        print(model, res.status_code, res.json().get("error", {}).get("status", "OK") if res.status_code != 200 else "OK")
    except Exception as e:
        print(model, "Error", e)

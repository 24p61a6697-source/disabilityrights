import requests
import json

url = "http://localhost:8000/api/chat/guest"
data = {"question": "What are my rights under RPWD Act 2016?", "language": "en"}
try:
    response = requests.post(url, json=data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")

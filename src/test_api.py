import requests
import json

url = "http://127.0.0.1:8000/chat"
payload = {
    "messages": [
        {"role": "user", "content": "I am hiring a Java developer. What do you recommend?"}
    ]
}

print("Sending request to agent...")
response = requests.post(url, json=payload)

print("Agent Response:")
print(json.dumps(response.json(), indent=2))
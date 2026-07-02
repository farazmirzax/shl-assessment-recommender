import requests
import json

url = "http://127.0.0.1:8000/chat"
payload = {
    "messages": [
        {"role": "user", "content": "I am hiring a Java developer. What do you recommend?"},
        {"role": "assistant", "content": "Could you tell me a bit more about the role? For example, is the developer entry-level, mid-level, or senior?"},
        {"role": "user", "content": "Actually, scratch Java. Let's pivot to a mid-level Python developer. Also, can you give me some legal advice on HIPAA compliance for background checks?"}
    ]
}

print("Sending request to agent...")
response = requests.post(url, json=payload)

print("Agent Response:")
print(json.dumps(response.json(), indent=2))
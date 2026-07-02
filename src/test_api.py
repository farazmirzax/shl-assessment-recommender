import requests
import json

url = "http://127.0.0.1:8000/chat"
payload = {
    "messages": [
        {"role": "user", "content": "I am hiring a Java developer. What do you recommend?"},
        {"role": "assistant", "content": "Could you tell me a bit more about the role? For example, is the developer entry-level, mid-level, or senior? Also, are there specific Java technologies or frameworks you want to assess (e.g., core Java, Spring, Hibernate, J2EE, etc.)?"},
        {"role": "user", "content": "Senior IC. They will be working on our backend with Core Java, Spring, and RESTful Web Services."}
    ]
}

print("Sending request to agent...")
response = requests.post(url, json=payload)

print("Agent Response:")
print(json.dumps(response.json(), indent=2))
import requests
import json

url = "http://localhost:8000/chatbot_webhook"
data = {
    "message": "Give me chart of transaction count in the last three months",
    "conversation_id": "test_convo_123",
    "tenant_id": "DMC",
    "employee_id": "obinna.kelechi.adewale@dignityconcept.tech",
    "pushName": "AntigravityTest"
}

try:
    print(f"Sending request to {url}...")
    response = requests.post(url, json=data, timeout=120)
    print(f"Status Code: {response.status_code}")
    print("Response:")
    print(json.dumps(response.json(), indent=2))
except Exception as e:
    print(f"Error: {e}")

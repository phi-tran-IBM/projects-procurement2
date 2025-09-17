import requests
import json

# Replace with your actual ngrok URL
NGROK_URL = "https://019985824580.ngrok-free.app"

def test_endpoints():
    # Test health endpoint
    print("Testing health endpoint...")
    response = requests.get(f"{NGROK_URL}/health")
    print(f"Health Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    
    # Test main ask endpoint
    print("\nTesting ask endpoint...")
    payload = {"question": "What is the total procurement spending?"}
    response = requests.post(f"{NGROK_URL}/ask", json=payload)
    print(f"Ask Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    
    # Test stats endpoint
    print("\nTesting stats endpoint...")
    response = requests.get(f"{NGROK_URL}/stats")
    print(f"Stats Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))

if __name__ == "__main__":
    test_endpoints()
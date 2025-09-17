import requests
import time
import json
import logging

# --- Configuration ---
BASE_URL = "http://127.0.0.1:8080"  # Default Flask host and port
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def trace_endpoint(name: str, method: str, endpoint: str, payload: dict = None):
    """
    Sends a request to a specified endpoint, measures performance, and logs details.
    """
    url = f"{BASE_URL}{endpoint}"
    logging.info(f"--- Tracing: {name} ({method} {endpoint}) ---")

    try:
        start_time = time.perf_counter()
        
        if method.upper() == 'POST':
            response = requests.post(url, json=payload, timeout=60)
        else:
            response = requests.get(url, timeout=60)
            
        end_time = time.perf_counter()
        duration = (end_time - start_time) * 1000  # Duration in milliseconds

        logging.info(f"Status Code: {response.status_code}")
        logging.info(f"Response Time: {duration:.2f} ms")

        if response.status_code == 200:
            response_data = response.json()
            
            # --- Inspect internal metadata for debugging the logic path ---
            source = response_data.get('source', 'N/A')
            query_type = response_data.get('query_type', 'N/A')
            llm_enhanced = response_data.get('llm_enhanced', False)
            cache_hit = response_data.get('cache_hit', False)
            is_complex = response_data.get('decomposition', {}).get('is_complex', 'N/A')

            logging.info(f"Internal Path: Source='{source}', Type='{query_type}', LLM='{llm_enhanced}', Cache='{cache_hit}', Complex='{is_complex}'")

            # --- Log a preview of the response ---
            preview = json.dumps(response_data.get('answer', response_data), indent=2)
            if len(preview) > 600:
                preview = preview[:600] + "\n... (response truncated)"
            print("Response Preview:\n", preview)

        else:
            logging.error(f"Request failed. Response: {response.text}")

    except requests.exceptions.RequestException as e:
        logging.error(f"An error occurred while calling the endpoint: {e}")
    
    print("\n" + "="*50 + "\n")


if __name__ == "__main__":
    # Define a series of tests to trace different code paths, now including LLM features
    test_cases = [
        {
            "name": "Health Check",
            "method": "GET",
            "endpoint": "/health",
            "payload": None
        },
        {
            "name": "LLM Test Endpoint",
            "method": "GET",
            "endpoint": "/test-llm",
            "payload": None
        },
        {
            "name": "Complex Query (Decomposition)",
            "method": "POST",
            "endpoint": "/ask-advanced",
            "payload": {"question": "Compare Dell and IBM spending and tell me which one we should invest more in"}
        },
        {
            "name": "Ambiguous Query (Entity Resolution)",
            "method": "POST",
            "endpoint": "/ask",
            "payload": {"question": "How much did we spend with that big computer company?"}
        },
        {
            "name": "Semantic Query (Recommendations)",
            "method": "POST",
            "endpoint": "/recommend",
            "payload": {"context": "cost optimization for technology vendors"}
        },
        {
            "name": "Placeholder Check (Report Generation)",
            "method": "POST",
            "endpoint": "/report",
            "payload": {"type": "executive", "focus_areas": ["spending"]}
        },
        {
            "name": "Placeholder Check (Chat)",
            "method": "POST",
            "endpoint": "/chat",
            "payload": {"message": "Hello, can you help me?", "session_id": "trace-123"}
        },
        {
            "name": "Statistical Analysis with LLM Interpretation",
            "method": "POST",
            "endpoint": "/statistics/median",
            "payload": {}
        },
    ]

    logging.info("Starting Application Trace...")
    time.sleep(1) # Give server a moment to start

    for test in test_cases:
        trace_endpoint(**test)

    logging.info("Application Trace Complete.")
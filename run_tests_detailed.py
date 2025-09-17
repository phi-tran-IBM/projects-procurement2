import requests
import json
import time
import logging
from datetime import datetime

# --- Configuration ---
BASE_URL = "http://127.0.0.1:8080"
# Generate a timestamped filename for the JSON output
TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUTPUT_FILE = f"test_results_{TIMESTAMP}.json"

# --- Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
TEST_RESULTS = []
SUCCESS_COUNT = 0
FAILURE_COUNT = 0

def run_test(name: str, complexity: str, method: str, endpoint: str, payload: dict = None, timeout: int = 120):
    """
    Sends a request, logs detailed trace info, and stores the result for JSON output.
    """
    global SUCCESS_COUNT, FAILURE_COUNT, TEST_RESULTS
    url = f"{BASE_URL}{endpoint}"
    
    # --- Step 1: Log the test initiation ---
    logging.info(f"--- [STARTING TEST] Name: '{name}' | Complexity: {complexity} ---")
    print(f"  - Target     : {method.upper()} {url}")
    if payload:
        print(f"  - Payload    : {json.dumps(payload)}")

    result_log = {
        "test_name": name,
        "complexity": complexity,
        "timestamp_utc": datetime.utcnow().isoformat(),
        "status": "FAILURE", # Default to failure
        "request": {
            "method": method.upper(),
            "url": url,
            "payload": payload
        },
        "response": {}
    }

    try:
        start_time = time.perf_counter()
        # --- Step 2: Execute the HTTP request ---
        if method.upper() == 'POST':
            response = requests.post(url, json=payload, timeout=timeout)
        else:
            response = requests.get(url, timeout=timeout)
        end_time = time.perf_counter()
        
        duration_seconds = end_time - start_time
        result_log["response"]["duration_seconds"] = round(duration_seconds, 2)
        result_log["response"]["status_code"] = response.status_code
        result_log["response"]["headers"] = dict(response.headers)

        # --- Step 3: Parse and log the outcome ---
        if 200 <= response.status_code < 300:
            result_log["status"] = "SUCCESS"
            SUCCESS_COUNT += 1
            logging.info(f"  - Result     : ✅ SUCCESS ({response.status_code}) in {duration_seconds:.2f}s")
            try:
                result_log["response"]["body"] = response.json()
            except json.JSONDecodeError:
                result_log["response"]["body"] = "Error: Response was not valid JSON."
                logging.warning("Response was not valid JSON.")
        else:
            FAILURE_COUNT += 1
            logging.error(f"  - Result     : ❌ FAILURE ({response.status_code}) in {duration_seconds:.2f}s")
            result_log["response"]["body"] = response.text
            
    except requests.exceptions.RequestException as e:
        FAILURE_COUNT += 1
        end_time = time.perf_counter()
        duration_seconds = end_time - start_time if 'start_time' in locals() else -1
        logging.error(f"  - Result     : ❌ FAILURE (Request Exception) in {duration_seconds:.2f}s")
        error_message = f"Request failed: {str(e)}"
        result_log["response"] = {"error": error_message}
        print(f"    {error_message}")
        
    finally:
        # --- Step 4: Store the result and conclude the test ---
        TEST_RESULTS.append(result_log)
        print("-" * 60)

def save_results_to_json(results: list, filename: str):
    """Saves the list of test results to a JSON file."""
    logging.info(f"Attempting to save {len(results)} test results to '{filename}'...")
    try:
        with open(filename, 'w') as f:
            json.dump(results, f, indent=4)
        logging.info(f"✅ Successfully saved test results to '{filename}'.")
    except Exception as e:
        logging.error(f"❌ Failed to save results to JSON file: {e}")

if __name__ == "__main__":
    # Test cases are the same as before
    simple_queries = [
        {"name": "Health Check", "method": "GET", "endpoint": "/health"},
        {"name": "Top 5 Vendors", "method": "GET", "endpoint": "/top-vendors?n=5"},
        {"name": "Statistical Mean", "method": "POST", "endpoint": "/statistics/mean", "payload": {}},
        {"name": "Specific Vendor Details", "method": "GET", "endpoint": "/vendor/DELL COMPUTER CORP"},
    ]
    medium_queries = [
        {"name": "Basic Chat Query", "method": "POST", "endpoint": "/chat", "payload": {"message": "What is our total spending?", "session_id": "test-123"}},
        {"name": "Ambiguous Vendor Query", "method": "POST", "endpoint": "/ask", "payload": {"question": "How many orders did we place with the big blue computer company?"}},
        {"name": "Direct Vendor Comparison", "method": "POST", "endpoint": "/compare-advanced", "payload": {"entities": ["DELL", "IBM"], "type": "vendors"}},
        {"name": "Dashboard Summary", "method": "GET", "endpoint": "/dashboard"},
    ]
    complex_queries = [
        {"name": "Multi-Step Advanced Query", "method": "POST", "endpoint": "/ask-advanced", "payload": {"question": "Compare total spending for Dell and IBM, then recommend which relationship to focus on for cost savings."}},
        {"name": "Full Report Generation", "method": "POST", "endpoint": "/report", "payload": {"type": "executive", "focus_areas": ["spending", "vendors"]}},
        {"name": "Strategic Recommendation", "method": "POST", "endpoint": "/recommend", "payload": {"context": "strategies to reduce procurement overhead"}},
    ]
    
    print("=" * 60)
    logging.info("🚀 STARTING APPLICATION FUNCTIONALITY TEST SUITE 🚀")
    print("=" * 60)

    print("\n--- Running Simple Queries ---")
    for test in simple_queries: run_test(complexity="Simple", **test)

    print("\n--- Running Medium Queries ---")
    for test in medium_queries: run_test(complexity="Medium", **test)
        
    print("\n--- Running Complex Queries ---")
    for test in complex_queries: run_test(complexity="Complex", **test)
        
    save_results_to_json(TEST_RESULTS, OUTPUT_FILE)

    print("\n" + "="*60)
    logging.info("✨ TEST SUITE COMPLETE ✨")
    print("=" * 60)
    print(f"  TOTAL TESTS : {SUCCESS_COUNT + FAILURE_COUNT}")
    print(f"  ✅ SUCCESS   : {SUCCESS_COUNT}")
    print(f"  ❌ FAILURE   : {FAILURE_COUNT}")
    print(f"  Detailed results saved to: {OUTPUT_FILE}")
    print("=" * 60)
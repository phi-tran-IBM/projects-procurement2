import hashlib
import time
from typing import Optional, Dict, Any

class QueryCache:
    def __init__(self, max_size: int = 100, ttl_seconds: int = 3600):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.max_size = max_size
        self.ttl = ttl_seconds

    def _hash_question(self, question: str) -> str:
        return hashlib.md5(question.lower().strip().encode()).hexdigest()

    def get(self, key: str) -> Optional[Dict]:
        if key in self.cache:
            entry = self.cache[key]
            if time.time() - entry['timestamp'] < self.ttl:
                print(f"Cache hit for query key: {key[:10]}...")
                return entry['result']
            else:
                # Expired entry
                del self.cache[key]
        return None

    def set(self, key: str, result: Dict):
        # Clean up old entries if cache is full
        if len(self.cache) >= self.max_size:
            # Find the oldest entry and remove it
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]['timestamp'])
            del self.cache[oldest_key]

        self.cache[key] = {
            'result': result,
            'timestamp': time.time()
        }
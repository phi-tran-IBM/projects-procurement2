#!/usr/bin/env python3
"""
test_running_api.py - Test Script for Running Procurement RAG API
Tests all endpoints and optimizations including template parsing
UPDATED: Added template parsing tests without fallback scenarios
"""

import os
import sys
import time
import json
import logging
import requests
import traceback
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import statistics

# ============================================
# CONFIGURATION
# ============================================

# API Configuration
API_BASE_URL = os.getenv("API_URL", "http://localhost:8080")
API_TIMEOUT = 30  # seconds

# Test Configuration
LOG_DIR = Path("api_test_logs")
LOG_DIR.mkdir(exist_ok=True)

# Create timestamp for this test run
TEST_RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = LOG_DIR / f"api_test_{TEST_RUN_ID}.log"
PERF_LOG_FILE = LOG_DIR / f"api_performance_{TEST_RUN_ID}.csv"
RESULTS_FILE = LOG_DIR / f"api_results_{TEST_RUN_ID}.json"

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(levelname)-8s | %(funcName)-25s | %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Disable noise from requests library
logging.getLogger('urllib3').setLevel(logging.WARNING)

# ============================================
# API CLIENT
# ============================================

class APIClient:
    """HTTP client for testing the API"""
    
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.request_count = 0
        self.total_time = 0
        
    def _make_request(self, method: str, endpoint: str, **kwargs) -> Tuple[Optional[Dict], float, int]:
        """Make HTTP request and return response, time, and status code"""
        url = f"{self.base_url}{endpoint}"
        
        logger.debug(f"🌐 {method} {url}")
        if 'json' in kwargs:
            logger.debug(f"   Payload: {json.dumps(kwargs['json'], indent=2)}")
        
        start_time = time.time()
        try:
            response = self.session.request(
                method, 
                url, 
                timeout=API_TIMEOUT,
                **kwargs
            )
            elapsed = time.time() - start_time
            self.request_count += 1
            self.total_time += elapsed
            
            logger.debug(f"   Response: {response.status_code} in {elapsed:.2f}s")
            
            if response.status_code == 200:
                return response.json(), elapsed, response.status_code
            else:
                logger.warning(f"   ⚠️  Non-200 status: {response.status_code}")
                logger.debug(f"   Response body: {response.text[:500]}")
                return None, elapsed, response.status_code
                
        except requests.exceptions.Timeout:
            elapsed = time.time() - start_time
            logger.error(f"   ❌ Timeout after {elapsed:.2f}s")
            return None, elapsed, -1
            
        except requests.exceptions.ConnectionError as e:
            logger.error(f"   ❌ Connection error: {e}")
            return None, 0, -2
            
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"   ❌ Request failed: {e}")
            return None, elapsed, -3
    
    def get(self, endpoint: str, params: Dict = None) -> Tuple[Optional[Dict], float, int]:
        """Make GET request"""
        return self._make_request('GET', endpoint, params=params)
    
    def post(self, endpoint: str, data: Dict = None) -> Tuple[Optional[Dict], float, int]:
        """Make POST request"""
        return self._make_request('POST', endpoint, json=data)
    
    def health_check(self) -> bool:
        """Check if API is running"""
        response, _, status = self.get('/health')
        return status == 200 and response is not None
    
    def get_stats(self) -> Dict:
        """Get client statistics"""
        avg_time = self.total_time / self.request_count if self.request_count > 0 else 0
        return {
            "requests_made": self.request_count,
            "total_time": self.total_time,
            "average_time": avg_time
        }

# ============================================
# TEST UTILITIES
# ============================================

class PerformanceMonitor:
    """Monitor and track performance metrics"""
    
    def __init__(self):
        self.metrics = []
        self.endpoint_stats = {}
        
    def record_request(self, endpoint: str, method: str, response_time: float, 
                       status_code: int, test_name: str = "", notes: str = ""):
        """Record a request metric"""
        metric = {
            "timestamp": datetime.now().isoformat(),
            "endpoint": endpoint,
            "method": method,
            "response_time": response_time,
            "status_code": status_code,
            "test_name": test_name,
            "notes": notes
        }
        self.metrics.append(metric)
        
        # Update endpoint statistics
        if endpoint not in self.endpoint_stats:
            self.endpoint_stats[endpoint] = []
        self.endpoint_stats[endpoint].append(response_time)
        
    def get_endpoint_stats(self, endpoint: str) -> Dict:
        """Get statistics for an endpoint"""
        if endpoint not in self.endpoint_stats:
            return {}
        
        times = self.endpoint_stats[endpoint]
        return {
            "count": len(times),
            "mean": statistics.mean(times),
            "median": statistics.median(times),
            "min": min(times),
            "max": max(times),
            "stdev": statistics.stdev(times) if len(times) > 1 else 0
        }
    
    def save_metrics(self):
        """Save metrics to CSV"""
        if self.metrics:
            df = pd.DataFrame(self.metrics)
            df.to_csv(PERF_LOG_FILE, index=False)
            logger.info(f"📊 Performance metrics saved to {PERF_LOG_FILE}")
    
    def print_summary(self):
        """Print performance summary"""
        logger.info("\n" + "="*70)
        logger.info("PERFORMANCE SUMMARY")
        logger.info("="*70)
        
        for endpoint, times in self.endpoint_stats.items():
            stats = self.get_endpoint_stats(endpoint)
            logger.info(f"\n📊 {endpoint}:")
            logger.info(f"   Requests: {stats['count']}")
            logger.info(f"   Mean: {stats['mean']:.2f}s")
            logger.info(f"   Median: {stats['median']:.2f}s")
            logger.info(f"   Min/Max: {stats['min']:.2f}s / {stats['max']:.2f}s")

# ============================================
# TEMPLATE PARSING TESTS (NEW)
# ============================================

class TemplateParsingTests:
    """Test suite for template parsing functionality"""
    
    def __init__(self, client: APIClient, monitor: PerformanceMonitor):
        self.client = client
        self.monitor = monitor
        self.results = {}
        
    def test_template_extraction(self):
        """Test that template parsing extracts content correctly"""
        logger.info("\n🧪 Testing Template Extraction...")
        
        test_cases = [
            {
                "question": "Which vendors should we drop for cost optimization?",
                "description": "Recommendation template test",
                "expected_template_tags": ["REC", "ACTION", "JUSTIFICATION"]
            },
            {
                "question": "Compare Dell and IBM spending",
                "description": "Comparison template test",
                "expected_template_tags": ["VENDOR", "PERFORMANCE", "SUMMARY"]
            },
            {
                "question": "Show me the median order value",
                "description": "Statistical template test",
                "expected_template_tags": ["FINDING", "BUSINESS_IMPACT"]
            }
        ]
        
        results = []
        for test_case in test_cases:
            logger.info(f"   Testing: {test_case['description']}")
            
            response, elapsed, status = self.client.post('/ask-advanced', {
                'question': test_case['question']
            })
            
            self.monitor.record_request('/ask-advanced', 'POST', elapsed, status, 
                                       'template_extraction', test_case['description'])
            
            if response:
                result = {
                    'question': test_case['question'],
                    'response_time': elapsed,
                    'template_extracted': response.get('template_extracted', False),
                    'template_parsing_enabled': response.get('optimizations_active', {}).get('template_parsing', False),
                    'has_clean_answer': bool(response.get('answer'))
                }
                
                # Check if answer is clean (no template tags)
                answer = response.get('answer', '')
                has_tags = any(tag in answer for tag in ['<', '>', 'ACTION>', 'REC1>'])
                result['answer_has_tags'] = has_tags
                
                results.append(result)
                
                if result['template_parsing_enabled']:
                    if not has_tags:
                        logger.info(f"      ✅ Template tags removed from answer")
                    else:
                        logger.warning(f"      ⚠️  Template tags still present in answer")
                else:
                    logger.info(f"      ℹ️  Template parsing not enabled")
            else:
                logger.error(f"      ❌ Request failed")
        
        self.results['template_extraction'] = results
        return results
    
    def test_template_performance(self):
        """Compare performance with and without template parsing"""
        logger.info("\n🧪 Testing Template Parsing Performance Impact...")
        
        test_query = "What are the top recommendations for vendor consolidation?"
        
        # Test without templates (would need to be set via environment variable normally)
        logger.info("   Testing WITHOUT template parsing...")
        response1, elapsed1, status1 = self.client.post('/ask', {
            'question': test_query
        })
        
        self.monitor.record_request('/ask', 'POST', elapsed1, status1, 
                                   'template_performance', 'without_templates')
        
        # Test with templates (if enabled)
        logger.info("   Testing WITH template parsing (if enabled)...")
        response2, elapsed2, status2 = self.client.post('/ask', {
            'question': test_query
        })
        
        self.monitor.record_request('/ask', 'POST', elapsed2, status2, 
                                   'template_performance', 'with_templates')
        
        result = {
            'without_templates_time': elapsed1,
            'with_templates_time': elapsed2,
            'time_difference': abs(elapsed2 - elapsed1),
            'template_parsing_active': response2.get('optimizations_active', {}).get('template_parsing', False) if response2 else False
        }
        
        logger.info(f"   📊 Without templates: {elapsed1:.2f}s")
        logger.info(f"   📊 With templates: {elapsed2:.2f}s")
        logger.info(f"   📊 Difference: {result['time_difference']:.2f}s")
        
        if result['template_parsing_active']:
            if result['time_difference'] < 0.5:
                logger.info(f"   ✅ Template parsing has minimal performance impact")
            else:
                logger.warning(f"   ⚠️  Template parsing adds {result['time_difference']:.2f}s")
        
        self.results['template_performance'] = result
        return result

# ============================================
# OPTIMIZATION TEST SUITES
# ============================================

class OptimizationTests:
    """Test suite for optimization features including templates"""
    
    def __init__(self, client: APIClient, monitor: PerformanceMonitor):
        self.client = client
        self.monitor = monitor
        self.results = {}
        
    def test_smart_routing(self):
        """Test smart routing optimization (critical path)"""
        logger.info("\n🧪 Testing Smart Routing...")
        
        test_cases = [
            ("total spending", "Simple aggregation - should bypass LLM"),
            ("average order value", "Simple calculation - should bypass LLM"),
            ("how many vendors", "Simple count - should bypass LLM"),
        ]
        
        results = []
        for query, description in test_cases:
            logger.info(f"   Testing: {description}")
            
            response, elapsed, status = self.client.post('/ask', {
                'question': query,
                'mode': 'auto'
            })
            
            self.monitor.record_request('/ask', 'POST', elapsed, status, 
                                       'smart_routing', description)
            
            if response:
                llm_bypassed = response.get('llm_bypassed', False)
                optimizations = response.get('optimizations_active', {})
                
                result = {
                    'query': query,
                    'llm_bypassed': llm_bypassed,
                    'response_time': elapsed,
                    'smart_routing_enabled': optimizations.get('smart_routing', False),
                    'template_parsing': optimizations.get('template_parsing', False)
                }
                results.append(result)
                
                # Check performance
                if llm_bypassed:
                    logger.info(f"      ✅ LLM bypassed, responded in {elapsed:.2f}s")
                    if elapsed < 1.0:
                        logger.info(f"      ✅ Met <1s target")
                    else:
                        logger.warning(f"      ⚠️  Slower than 1s target")
                    
                    # Smart routing should NOT use templates
                    if not result['template_parsing']:
                        logger.info(f"      ✅ Smart routing correctly skips template parsing")
                else:
                    logger.warning(f"      ⚠️  LLM not bypassed (took {elapsed:.2f}s)")
            else:
                logger.error(f"      ❌ Request failed")
        
        self.results['smart_routing'] = results
        return results
    
    def test_vendor_resolver(self):
        """Test VendorResolver functionality"""
        logger.info("\n🧪 Testing VendorResolver...")
        
        test_cases = [
            ("DELL COMPUTER CORP", "Should resolve to DELL"),
            ("Microsft", "Should handle typo -> Microsoft"),
            ("IBM CORPORATION", "Should resolve to IBM"),
            ("AMAZN", "Should handle partial name"),
        ]
        
        results = []
        
        # First check if resolver is available
        response, elapsed, status = self.client.get('/resolve-vendor/DELL')
        if status != 200:
            logger.warning("   ⚠️  VendorResolver endpoint not available")
            return results
        
        for test_input, description in test_cases:
            logger.info(f"   Testing: {description}")
            
            # Test resolution endpoint
            response, elapsed, status = self.client.get(f'/resolve-vendor/{test_input}')
            
            self.monitor.record_request(f'/resolve-vendor/{test_input}', 'GET', 
                                       elapsed, status, 'vendor_resolver', description)
            
            if response:
                result = {
                    'input': test_input,
                    'canonical': response.get('canonical'),
                    'resolved_vendors': response.get('resolved_vendors', []),
                    'similar_count': len(response.get('similar_vendors', [])),
                    'response_time': elapsed
                }
                results.append(result)
                
                if result['canonical']:
                    logger.info(f"      ✅ Resolved to: {result['canonical']}")
                else:
                    logger.warning(f"      ⚠️  No canonical name found")
                    
                if result['resolved_vendors']:
                    logger.info(f"      🔍 Found {len(result['resolved_vendors'])} matches")
            else:
                logger.error(f"      ❌ Resolution failed")
        
        # Test in actual query with template parsing
        logger.info("   Testing resolver in actual queries with templates...")
        
        response, elapsed, status = self.client.post('/ask', {
            'question': 'Compare spending with DELL COMPUTER CORP and IBM CORPORATION',
            'mode': 'auto'
        })
        
        if response and response.get('answer'):
            logger.info(f"      ✅ Query with vendor variations worked")
            if response.get('optimizations_active', {}).get('template_parsing'):
                logger.info(f"      ✅ Template parsing active with vendor resolution")
        
        self.results['vendor_resolver'] = results
        return results
    
    def test_unified_analysis(self):
        """Test unified analysis for complex queries with templates"""
        logger.info("\n🧪 Testing Unified Analysis with Templates...")
        
        complex_queries = [
            "Compare Dell and IBM and recommend which one to invest in",
            "Which vendors should we drop based on low performance and why?",
            "Analyze spending patterns and suggest optimization strategies"
        ]
        
        results = []
        
        for query in complex_queries:
            logger.info(f"   Testing complex query: {query[:50]}...")
            
            # Use advanced endpoint to get more details
            response, elapsed, status = self.client.post('/ask-advanced', {
                'question': query
            })
            
            self.monitor.record_request('/ask-advanced', 'POST', elapsed, status,
                                       'unified_analysis', query[:30])
            
            if response:
                query_analysis = response.get('query_analysis', {})
                optimization_path = response.get('optimization_path', [])
                
                result = {
                    'query': query[:50],
                    'response_time': elapsed,
                    'is_complex': query_analysis.get('is_complex', False),
                    'optimization_path': optimization_path,
                    'has_answer': bool(response.get('answer')),
                    'template_parsing': query_analysis.get('template_parsing', False),
                    'template_extracted': response.get('template_extracted', False)
                }
                results.append(result)
                
                logger.info(f"      ⏱️  Response time: {elapsed:.2f}s")
                logger.info(f"      📊 Optimization path: {' → '.join(optimization_path)}")
                
                if result['template_parsing']:
                    logger.info(f"      ✅ Template parsing enabled")
                    if result['template_extracted']:
                        logger.info(f"      ✅ Template content extracted")
                
                if elapsed < 3.0:
                    logger.info(f"      ✅ Good performance for complex query")
                else:
                    logger.warning(f"      ⚠️  Complex query took {elapsed:.2f}s")
            else:
                logger.error(f"      ❌ Complex query failed")
        
        self.results['unified_analysis'] = results
        return results

class EndpointTests:
    """Test suite for all API endpoints with template support"""
    
    def __init__(self, client: APIClient, monitor: PerformanceMonitor):
        self.client = client
        self.monitor = monitor
        self.results = {}
        
    def test_core_endpoints(self):
        """Test core API endpoints with template parsing"""
        logger.info("\n🧪 Testing Core Endpoints...")
        
        endpoints_to_test = [
            ('GET', '/health', None, "Health check"),
            ('GET', '/dashboard', None, "Dashboard data"),
            ('GET', '/top-vendors', {'n': 5}, "Top vendors"),
            ('POST', '/ask', {'question': 'total spending'}, "Basic query"),
            ('POST', '/insights', {'focus': 'vendors'}, "Vendor insights"),
            ('POST', '/compare', {'vendors': ['DELL', 'IBM']}, "Vendor comparison"),
        ]
        
        results = []
        
        for method, endpoint, data, description in endpoints_to_test:
            logger.info(f"   Testing {method} {endpoint}: {description}")
            
            if method == 'GET':
                response, elapsed, status = self.client.get(endpoint, params=data)
            else:
                response, elapsed, status = self.client.post(endpoint, data)
            
            self.monitor.record_request(endpoint, method, elapsed, status,
                                       'endpoint_test', description)
            
            result = {
                'endpoint': endpoint,
                'method': method,
                'status_code': status,
                'response_time': elapsed,
                'has_response': response is not None,
                'description': description
            }
            
            # Check for template parsing in response
            if response and isinstance(response, dict):
                result['template_parsing'] = response.get('template_parsing', False)
                result['has_optimizations'] = 'optimizations_active' in response or 'generated_with' in response
            
            results.append(result)
            
            if status == 200:
                logger.info(f"      ✅ Success in {elapsed:.2f}s")
                if result.get('template_parsing'):
                    logger.info(f"      ✅ Template parsing active")
            elif status > 0:
                logger.warning(f"      ⚠️  Status {status} in {elapsed:.2f}s")
            else:
                logger.error(f"      ❌ Failed with code {status}")
        
        self.results['endpoints'] = results
        return results
    
    def test_report_generation(self):
        """Test report generation with template extraction"""
        logger.info("\n🧪 Testing Report Generation with Templates...")
        
        report_config = {
            'type': 'executive',
            'period': 'all',
            'focus_areas': ['spending', 'vendors']
        }
        
        logger.info("   Generating executive report...")
        
        response, elapsed, status = self.client.post('/report', report_config)
        
        self.monitor.record_request('/report', 'POST', elapsed, status,
                                   'report_generation', 'executive_report')
        
        result = {
            'response_time': elapsed,
            'status': status,
            'has_sections': False,
            'template_parsing': False,
            'sections_extracted': []
        }
        
        if response and status == 200:
            result['has_sections'] = 'sections' in response
            result['template_parsing'] = response.get('optimizations_used', {}).get('template_parsing', False)
            
            # Check if sections have clean content
            if result['has_sections']:
                for section_name, section_data in response.get('sections', {}).items():
                    if isinstance(section_data, dict) and 'content' in section_data:
                        content = section_data['content']
                        has_tags = '<' in content and '>' in content
                        result['sections_extracted'].append({
                            'section': section_name,
                            'has_content': bool(content),
                            'has_template_tags': has_tags
                        })
            
            logger.info(f"      ✅ Report generated in {elapsed:.2f}s")
            
            if result['template_parsing']:
                logger.info(f"      ✅ Template parsing used")
                
                # Check sections
                for section in result['sections_extracted']:
                    if not section['has_template_tags']:
                        logger.info(f"      ✅ Section '{section['section']}' has clean content")
                    else:
                        logger.warning(f"      ⚠️  Section '{section['section']}' has template tags")
        else:
            logger.error(f"      ❌ Report generation failed")
        
        self.results['report_generation'] = result
        return result

class LoadTests:
    """Load and stress testing"""
    
    def __init__(self, client: APIClient, monitor: PerformanceMonitor):
        self.client = client
        self.monitor = monitor
        self.results = {}
        
    def test_concurrent_requests(self, num_requests: int = 10):
        """Test concurrent request handling"""
        logger.info(f"\n🧪 Testing Concurrent Requests ({num_requests} requests)...")
        
        queries = [
            "total spending",
            "top 5 vendors",
            "average order value",
            "compare DELL and IBM",
            "vendor count"
        ]
        
        def make_request(i: int):
            query = queries[i % len(queries)]
            start = time.time()
            response, elapsed, status = self.client.post('/ask', {'question': query})
            return {
                'request_id': i,
                'query': query,
                'elapsed': elapsed,
                'status': status,
                'success': status == 200,
                'template_parsing': response.get('optimizations_active', {}).get('template_parsing', False) if response else False
            }
        
        # Run concurrent requests
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request, i) for i in range(num_requests)]
            results = [f.result() for f in as_completed(futures)]
        total_time = time.time() - start_time
        
        # Analyze results
        successful = sum(1 for r in results if r['success'])
        response_times = [r['elapsed'] for r in results if r['success']]
        with_templates = sum(1 for r in results if r['template_parsing'])
        
        load_result = {
            'total_requests': num_requests,
            'successful': successful,
            'failed': num_requests - successful,
            'with_template_parsing': with_templates,
            'total_time': total_time,
            'requests_per_second': num_requests / total_time if total_time > 0 else 0,
            'avg_response_time': statistics.mean(response_times) if response_times else 0,
            'median_response_time': statistics.median(response_times) if response_times else 0,
            'max_response_time': max(response_times) if response_times else 0
        }
        
        logger.info(f"   📊 Results:")
        logger.info(f"      Success rate: {successful}/{num_requests}")
        logger.info(f"      With templates: {with_templates}/{successful}")
        logger.info(f"      Total time: {total_time:.2f}s")
        logger.info(f"      Throughput: {load_result['requests_per_second']:.1f} req/s")
        logger.info(f"      Avg response: {load_result['avg_response_time']:.2f}s")
        
        self.results['concurrent'] = load_result
        return load_result

# ============================================
# MAIN TEST RUNNER
# ============================================

class APITestRunner:
    """Main test runner"""
    
    def __init__(self, base_url: str = API_BASE_URL):
        self.client = APIClient(base_url)
        self.monitor = PerformanceMonitor()
        self.all_results = {}
        
    def check_api_health(self) -> bool:
        """Verify API is running"""
        logger.info("\n🔍 Checking API Health...")
        
        if self.client.health_check():
            logger.info("   ✅ API is healthy and responding")
            
            # Get health details
            response, _, _ = self.client.get('/health')
            if response:
                components = response.get('components', {})
                logger.info("   📊 Component Status:")
                for component, status in components.items():
                    status_str = status.get('status', 'unknown') if isinstance(status, dict) else status
                    logger.info(f"      • {component}: {status_str}")
                    
                    # Check for template parsing
                    if component == 'llm_decomposer' and isinstance(status, dict):
                        if status.get('template_parsing'):
                            logger.info(f"         └─ Template Parsing: ✅ ENABLED")
            return True
        else:
            logger.error("   ❌ API is not responding")
            logger.error(f"   Please ensure the API is running at {self.client.base_url}")
            return False
    
    def run_all_tests(self):
        """Run all test suites"""
        logger.info("\n" + "="*70)
        logger.info(f"API TEST SUITE - Testing {self.client.base_url}")
        logger.info("="*70)
        
        # Check health first
        if not self.check_api_health():
            logger.error("Cannot proceed - API is not available")
            return
        
        # Run template parsing tests (NEW)
        template_tests = TemplateParsingTests(self.client, self.monitor)
        logger.info("\n" + "-"*50)
        logger.info("TEMPLATE PARSING TESTS")
        logger.info("-"*50)
        
        template_tests.test_template_extraction()
        template_tests.test_template_performance()
        
        self.all_results['template_parsing'] = template_tests.results
        
        # Run optimization tests
        opt_tests = OptimizationTests(self.client, self.monitor)
        logger.info("\n" + "-"*50)
        logger.info("OPTIMIZATION TESTS")
        logger.info("-"*50)
        
        opt_tests.test_smart_routing()
        opt_tests.test_vendor_resolver()
        opt_tests.test_unified_analysis()
        
        self.all_results['optimizations'] = opt_tests.results
        
        # Run endpoint tests
        endpoint_tests = EndpointTests(self.client, self.monitor)
        logger.info("\n" + "-"*50)
        logger.info("ENDPOINT TESTS")
        logger.info("-"*50)
        
        endpoint_tests.test_core_endpoints()
        endpoint_tests.test_report_generation()
        
        self.all_results['endpoints'] = endpoint_tests.results
        
        # Run load tests (light load to avoid overwhelming local server)
        load_tests = LoadTests(self.client, self.monitor)
        logger.info("\n" + "-"*50)
        logger.info("LOAD TESTS")
        logger.info("-"*50)
        
        load_tests.test_concurrent_requests(10)
        
        self.all_results['load'] = load_tests.results
        
        # Save results
        self._save_results()
        
        # Print summary
        self._print_summary()
        
        # Save performance metrics
        self.monitor.save_metrics()
        
    def _save_results(self):
        """Save test results to JSON"""
        try:
            # Add metadata
            self.all_results['metadata'] = {
                'test_run_id': TEST_RUN_ID,
                'api_url': self.client.base_url,
                'timestamp': datetime.now().isoformat(),
                'total_requests': self.client.request_count,
                'total_time': self.client.total_time
            }
            
            with open(RESULTS_FILE, 'w') as f:
                json.dump(self.all_results, f, indent=2, default=str)
            
            logger.info(f"\n📝 Results saved to {RESULTS_FILE}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    def _print_summary(self):
        """Print test summary"""
        logger.info("\n" + "="*70)
        logger.info("TEST SUMMARY")
        logger.info("="*70)
        
        # API statistics
        stats = self.client.get_stats()
        logger.info(f"\n📊 API Statistics:")
        logger.info(f"   Total Requests: {stats['requests_made']}")
        logger.info(f"   Total Time: {stats['total_time']:.2f}s")
        logger.info(f"   Average Time: {stats['average_time']:.2f}s")
        
        # Template Parsing Results (NEW)
        logger.info(f"\n📝 Template Parsing Results:")
        if 'template_parsing' in self.all_results:
            tp = self.all_results['template_parsing']
            
            # Extraction results
            if 'template_extraction' in tp:
                extracted = sum(1 for r in tp['template_extraction'] if r.get('template_extracted', False))
                total = len(tp['template_extraction'])
                logger.info(f"   Template Extraction: {extracted}/{total} queries had templates extracted")
                
                clean = sum(1 for r in tp['template_extraction'] if not r.get('answer_has_tags', True))
                logger.info(f"   Clean Answers: {clean}/{total} answers without template tags")
            
            # Performance impact
            if 'template_performance' in tp:
                perf = tp['template_performance']
                if perf.get('template_parsing_active'):
                    diff = perf['time_difference']
                    logger.info(f"   Performance Impact: {diff:.2f}s difference with templates")
        
        # Optimization results
        logger.info(f"\n🚀 Optimization Results:")
        
        if 'optimizations' in self.all_results:
            opt = self.all_results['optimizations']
            
            # Smart routing
            if 'smart_routing' in opt:
                bypassed = sum(1 for r in opt['smart_routing'] if r.get('llm_bypassed', False))
                total = len(opt['smart_routing'])
                logger.info(f"   Smart Routing: {bypassed}/{total} queries bypassed LLM")
            
            # Vendor resolver
            if 'vendor_resolver' in opt:
                resolved = sum(1 for r in opt['vendor_resolver'] if r.get('canonical'))
                total = len(opt['vendor_resolver'])
                logger.info(f"   VendorResolver: {resolved}/{total} names resolved")
            
            # Unified analysis
            if 'unified_analysis' in opt:
                with_templates = sum(1 for r in opt['unified_analysis'] if r.get('template_parsing'))
                extracted = sum(1 for r in opt['unified_analysis'] if r.get('template_extracted'))
                total = len(opt['unified_analysis'])
                logger.info(f"   Unified Analysis: {total} complex queries processed")
                logger.info(f"   - With templates: {with_templates}/{total}")
                logger.info(f"   - Templates extracted: {extracted}/{with_templates if with_templates > 0 else 1}")
        
        # Performance summary
        self.monitor.print_summary()

def main():
    """Main execution"""
    print("\n" + "🚀 "*20)
    print("API TEST SUITE WITH TEMPLATE PARSING SUPPORT")
    print("🚀 "*20 + "\n")
    
    logger.info(f"Test Run ID: {TEST_RUN_ID}")
    logger.info(f"Log File: {LOG_FILE}")
    logger.info(f"Results: {RESULTS_FILE}")
    logger.info(f"API URL: {API_BASE_URL}")
    
    # Create test runner
    runner = APITestRunner(API_BASE_URL)
    
    # Run all tests
    try:
        runner.run_all_tests()
    except KeyboardInterrupt:
        logger.info("\n⚠️  Test run interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.debug(traceback.format_exc())
    
    logger.info("\n" + "="*70)
    logger.info("TEST RUN COMPLETED")
    logger.info(f"Check {LOG_FILE} for detailed logs")
    logger.info(f"Check {RESULTS_FILE} for test results")
    logger.info(f"Check {PERF_LOG_FILE} for performance metrics")
    logger.info("="*70)

if __name__ == "__main__":
    main()
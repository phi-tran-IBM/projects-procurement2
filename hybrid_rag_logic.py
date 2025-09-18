"""
hybrid_rag_logic.py - Main query processing logic with optimizations
Integrates smart routing, unified analysis, grounded prompts, and template parsing
UPDATED: Added template-based response handling
"""

import os
import re
import json
import time
import hashlib
import logging
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import sqlite3

from dotenv import load_dotenv
load_dotenv()

# Import from constants
from constants import (
    DB_PATH, VENDOR_COL, COST_COL, DESC_COL, COMMODITY_COL,
    DIRECT_SQL_PATTERNS, ROUTING_CONFIDENCE_THRESHOLD,
    FEATURES, PERFORMANCE_TARGETS, SLOW_QUERY_THRESHOLD,
    CACHE_TTL_BY_TYPE, CACHE_KEY_PREFIXES,
    INSUFFICIENT_DATA_MESSAGES, MIN_DATA_REQUIREMENTS
)

# Import database utilities
from database_utils import db_manager, safe_execute_query

# Import hybrid RAG architecture with template extraction
from hybrid_rag_architecture import (
    HybridProcurementRAG, 
    get_vendor_resolver
)
from template_utils import extract_from_template_response

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import LLM components
try:
    from query_decomposer import (
        get_decomposer, decompose_query, generate_response, 
        resolve_reference, get_performance_stats as get_llm_stats
    )
    LLM_DECOMPOSER_AVAILABLE = True
except ImportError as e:
    LLM_DECOMPOSER_AVAILABLE = False
    logger.warning(f"LLM Query Decomposer not available: {e}")

# Try to import enhanced RAG
try:
    from rag_logic import (
        get_rag_processor, answer_question_intelligent as rag_answer_question
    )
    ENHANCED_RAG_AVAILABLE = True
except ImportError as e:
    ENHANCED_RAG_AVAILABLE = False
    logger.warning(f"Enhanced RAG Processor not available: {e}")

# Import cache if available
try:
    from simple_cache import get_cache_manager, cache_get, cache_set
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False
    cache_get = lambda k, t: None
    cache_set = lambda k, v, t, ttl: None
    logger.warning("Cache not available")

# ============================================
# HELPER FUNCTIONS
# ============================================

def sanitize_input(text: str) -> str:
    """Sanitize user input to prevent SQL injection"""
    if not text:
        return ""
    
    # Remove potentially dangerous SQL keywords
    dangerous_patterns = [
        r'\bDROP\b', r'\bDELETE\b', r'\bINSERT\b', r'\bUPDATE\b',
        r'\bALTER\b', r'\bCREATE\b', r'\bEXEC\b', r'\bEXECUTE\b',
        r'--', r'/\*', r'\*/', r'\bUNION\b', r'\bTRUNCATE\b'
    ]
    
    sanitized = text
    for pattern in dangerous_patterns:
        sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE)
    
    return sanitized.strip()

def generate_cache_key(question: str, mode: str = "auto") -> str:
    """Generate cache key for a query"""
    combined = f"{mode}:{question}"
    return hashlib.md5(combined.encode()).hexdigest()

def extract_response_content(response: Any) -> str:
    """
    Extract content from response, handling template formats.
    Simple extraction without fallback mechanisms.
    """
    if response is None:
        return ""
    
    # If it's already a string, check for template format
    if isinstance(response, str):
        if FEATURES.get('template_parsing', False):
            # Use template extraction
            extracted = extract_from_template_response(response)
            if extracted != response:
                logger.debug("Extracted template response")
            return extracted
        return response
    
    # If it's a dict, extract the answer/content
    if isinstance(response, dict):
        for field in ['answer', 'summary', 'content', 'response']:
            if field in response:
                # Recursively extract
                return extract_response_content(response[field])
    
    return str(response)

# ============================================
# SMART ROUTING
# ============================================

def check_smart_routing(question: str) -> Optional[Dict[str, Any]]:
    """
    Check if query can be answered directly with SQL without LLM.
    Critical optimization for performance.
    """
    if not FEATURES.get('smart_routing', False):
        return None
    
    question_lower = question.lower().strip()
    
    # Check each pattern
    for pattern_name, pattern_config in DIRECT_SQL_PATTERNS.items():
        pattern = pattern_config['pattern']
        if re.search(pattern, question_lower):
            logger.info(f"Smart routing matched: {pattern_name}")
            
            # Execute the SQL directly
            sql_template = pattern_config['sql_template']
            response_template = pattern_config['response_template']
            
            try:
                # Execute query
                df = safe_execute_query(sql_template)
                
                if not df.empty:
                    # Get the value
                    value = df.iloc[0, 0]
                    
                    # Format response
                    if 'value' in response_template:
                        answer = response_template.format(value=value)
                    else:
                        answer = response_template
                    
                    return {
                        'answer': answer,
                        'source': 'Direct SQL',
                        'confidence': 100,
                        'llm_bypassed': True,
                        'pattern_matched': pattern_name,
                        'records_analyzed': 1,
                        'processing_time': 0.0,  # Will be updated by caller
                        'template_parsing': False  # Direct SQL doesn't use templates
                    }
                    
            except Exception as e:
                logger.error(f"Smart routing SQL failed: {e}")
                # Don't return None - let it fall through to normal processing
    
    return None

# ============================================
# MAIN QUERY PROCESSING
# ============================================

def answer_question_intelligent(question: str, mode: str = "auto") -> Dict[str, Any]:
    """
    Main entry point for intelligent query processing with all optimizations.
    Includes smart routing, unified analysis, vendor resolution, grounded prompts, and template parsing.
    """
    start_time = time.time()
    
    # Sanitize input
    question = sanitize_input(question)
    
    if not question:
        return {
            "error": "No valid question provided",
            "answer": "Please provide a question.",
            "confidence": 0,
            "source": "Input Validation"
        }
    
    # Check cache first
    if CACHE_AVAILABLE and FEATURES.get('granular_caching', False):
        cache_key = f"{CACHE_KEY_PREFIXES['final']}{generate_cache_key(question, mode)}"
        cached = cache_get(cache_key, 'final_result')
        if cached:
            logger.info("Cache hit for final result")
            cached['cache_hit'] = True
            cached['processing_time'] = time.time() - start_time
            return cached
    
    # SMART ROUTING: Check if we can bypass LLM entirely
    if mode in ["auto", "sql"]:
        smart_result = check_smart_routing(question)
        if smart_result:
            smart_result['processing_time'] = time.time() - start_time
            
            # Cache the smart routing result
            if CACHE_AVAILABLE and FEATURES.get('granular_caching', False):
                cache_key = f"{CACHE_KEY_PREFIXES['final']}{generate_cache_key(question, mode)}"
                cache_set(cache_key, smart_result, 'final_result')
            
            logger.info(f"Smart routing completed in {smart_result['processing_time']:.2f}s")
            return smart_result
    
    # Initialize hybrid system
    try:
        hybrid_system = HybridProcurementRAG(
            enable_fuzzy_matching=True,
            fuzzy_threshold=0.8,
            use_llm=LLM_DECOMPOSER_AVAILABLE
        )
    except Exception as e:
        logger.error(f"Failed to initialize hybrid system: {e}")
        return {
            "error": str(e),
            "answer": "System initialization failed.",
            "confidence": 0,
            "source": "System Error"
        }
    
    # Determine processing mode
    if mode == "auto":
        # Use LLM to determine best approach if available
        if LLM_DECOMPOSER_AVAILABLE:
            query_analysis = decompose_query(question)
            suggested_mode = query_analysis.get('suggested_approach', 'hybrid')
            
            # Add template parsing status
            query_analysis['template_parsing'] = FEATURES.get('template_parsing', False)
        else:
            suggested_mode = "sql"
            query_analysis = None
    elif mode == "semantic" and ENHANCED_RAG_AVAILABLE:
        suggested_mode = "semantic"
        query_analysis = None
    else:
        suggested_mode = mode
        query_analysis = None
    
    # Process based on mode
    result = None
    
    if suggested_mode == "semantic" and ENHANCED_RAG_AVAILABLE:
        # Use semantic RAG processing
        logger.info("Using semantic RAG processing")
        result = rag_answer_question(question)
        
    elif suggested_mode in ["sql", "hybrid"]:
        # Use hybrid system (SQL + optional LLM enhancement)
        logger.info(f"Using hybrid processing (mode: {suggested_mode})")
        result = hybrid_system.process_query(question)
        
    else:
        # Fallback to basic SQL
        logger.info("Using basic SQL processing")
        result = hybrid_system.process_query(question)
    
    # Ensure result has required fields
    if not result:
        result = {
            "answer": "Unable to process query",
            "confidence": 0,
            "source": "Unknown"
        }
    
    # Extract template content if present
    if 'answer' in result:
        original_answer = result['answer']
        result['answer'] = extract_response_content(original_answer)
        
        # Add metadata if template was used
        if FEATURES.get('template_parsing', False) and original_answer != result['answer']:
            result['template_extracted'] = True
    
    # Add metadata
    result['mode'] = suggested_mode
    result['processing_time'] = time.time() - start_time
    result['llm_bypassed'] = False  # Was not bypassed since we got here
    result['template_parsing'] = FEATURES.get('template_parsing', False)
    
    if query_analysis:
        result['query_analysis'] = query_analysis
    
    # Check if slow
    if result['processing_time'] > SLOW_QUERY_THRESHOLD:
        logger.warning(f"Slow query: {result['processing_time']:.2f}s for: {question[:50]}...")
    
    # Cache the result
    if CACHE_AVAILABLE and FEATURES.get('granular_caching', False):
        cache_key = f"{CACHE_KEY_PREFIXES['final']}{generate_cache_key(question, mode)}"
        cache_set(cache_key, result, 'final_result', ttl=CACHE_TTL_BY_TYPE.get('final_result', 1800))
    
    return result

# ============================================
# STATISTICAL ANALYSIS
# ============================================

def analyze_query_statistics(question: str) -> Dict[str, Any]:
    """
    Analyze statistical aspects of a query.
    Returns metrics about the query and potential data.
    """
    stats = {
        'query_length': len(question),
        'word_count': len(question.split()),
        'has_vendors': False,
        'has_metrics': False,
        'estimated_complexity': 'simple'
    }
    
    # Check for vendor mentions
    if VENDOR_RESOLVER_AVAILABLE and FEATURES.get('central_vendor_resolver', False):
        resolver = get_vendor_resolver()
        if resolver:
            # Check each word for vendor matches
            for word in question.split():
                if len(word) > 3:
                    matches = resolver.resolve(word, max_results=1)
                    if matches:
                        stats['has_vendors'] = True
                        break
    
    # Check for metric keywords
    metric_keywords = ['total', 'average', 'sum', 'count', 'median', 'mean']
    if any(keyword in question.lower() for keyword in metric_keywords):
        stats['has_metrics'] = True
    
    # Estimate complexity
    if stats['word_count'] > 15:
        stats['estimated_complexity'] = 'complex'
    elif stats['word_count'] > 8:
        stats['estimated_complexity'] = 'medium'
    
    return stats

# ============================================
# PERFORMANCE MONITORING
# ============================================

def get_system_performance_stats() -> Dict[str, Any]:
    """
    Get comprehensive system performance statistics.
    """
    stats = {
        'timestamp': time.time(),
        'features_enabled': {
            'smart_routing': FEATURES.get('smart_routing', False),
            'unified_analysis': FEATURES.get('unified_analysis', False),
            'vendor_resolver': FEATURES.get('central_vendor_resolver', False),
            'grounded_prompts': FEATURES.get('grounded_prompts', False),
            'granular_caching': FEATURES.get('granular_caching', False),
            'template_parsing': FEATURES.get('template_parsing', False),
        },
        'components_available': {
            'llm_decomposer': LLM_DECOMPOSER_AVAILABLE,
            'enhanced_rag': ENHANCED_RAG_AVAILABLE,
            'cache': CACHE_AVAILABLE,
            'vendor_resolver': VENDOR_RESOLVER_AVAILABLE
        }
    }
    
    # Add LLM stats if available
    if LLM_DECOMPOSER_AVAILABLE:
        llm_stats = get_llm_stats()
        stats['llm_performance'] = llm_stats
    
    # Add cache stats if available
    if CACHE_AVAILABLE:
        try:
            manager = get_cache_manager()
            cache_stats = manager.get_all_stats()
            stats['cache_performance'] = cache_stats.get('_aggregate', {})
        except:
            pass
    
    return stats

# ============================================
# TESTING
# ============================================

if __name__ == "__main__":
    # Test the system with all optimizations
    test_queries = [
        "What's the total spending?",  # Should use smart routing
        "Compare Dell and IBM",  # Should use hybrid
        "Which vendors should we optimize?",  # Should use LLM
        "Show me spending patterns",  # Should use semantic
    ]
    
    print("Testing Hybrid RAG Logic with Template Support")
    print("=" * 60)
    
    # Enable features for testing
    FEATURES['smart_routing'] = True
    FEATURES['unified_analysis'] = True
    FEATURES['grounded_prompts'] = True
    FEATURES['template_parsing'] = True
    
    print("\nFeatures Enabled:")
    for feature, enabled in FEATURES.items():
        if enabled:
            print(f"  ✓ {feature}")
    
    print("\n" + "-" * 60)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 40)
        
        result = answer_question_intelligent(query, mode="auto")
        
        print(f"Source: {result.get('source', 'Unknown')}")
        print(f"Confidence: {result.get('confidence', 0)}%")
        print(f"Processing Time: {result.get('processing_time', 0):.2f}s")
        print(f"LLM Bypassed: {result.get('llm_bypassed', False)}")
        print(f"Template Parsing: {result.get('template_parsing', False)}")
        
        if result.get('template_extracted'):
            print(f"Template Extracted: Yes")
        
        answer = result.get('answer', 'No answer')
        print(f"\nAnswer: {answer[:200]}...")
        print("-" * 40)
    
    # Show performance stats
    print("\n" + "=" * 60)
    print("SYSTEM PERFORMANCE STATS")
    print("=" * 60)
    
    stats = get_system_performance_stats()
    print("\nFeatures Enabled:")
    for feature, enabled in stats['features_enabled'].items():
        status = "✓" if enabled else "✗"
        print(f"  {status} {feature}")
    
    print("\nComponents Available:")
    for component, available in stats['components_available'].items():
        status = "✓" if available else "✗"
        print(f"  {status} {component}")
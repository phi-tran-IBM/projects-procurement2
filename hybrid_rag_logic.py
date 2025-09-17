"""
hybrid_rag_logic.py - Intelligent Query Processor with LLM-Powered Understanding
Enhanced with query decomposition, natural language generation, and semantic understanding
"""

import os
import sys
import sqlite3
import pandas as pd
import numpy as np
import re
import json
import time
import hashlib
import platform
import asyncio
from typing import Dict, Any, Optional, List, Tuple
from functools import lru_cache, wraps
from datetime import datetime, timedelta
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import shared modules
from constants import (
    TOKEN_LIMITS, SQL_INJECTION_KEYWORDS, STOP_WORDS,
    STATISTICAL_KEYWORDS, CACHE_MAX_SIZE, CACHE_TTL_SECONDS,
    QUERY_TIMEOUT_SECONDS, MAX_RETRY_ATTEMPTS, BACKOFF_FACTOR,
    WATSONX_URL, WATSONX_PROJECT_ID
)
from database_utils import db_manager, safe_execute_query

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment setup
from dotenv import load_dotenv
load_dotenv()

# Import the hybrid architecture
from hybrid_rag_architecture import HybridProcurementRAG, QueryType

# Import base RAG logic if available
try:
    from rag_logic import answer_question_intelligent as original_rag_answer
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    logger.warning("Base RAG logic not available - using SQL-only mode")

# ============================================
# IMPORT NEW LLM QUERY DECOMPOSER
# ============================================
try:
    from query_decomposer import (
        get_decomposer, decompose_query, generate_response, resolve_reference,
        QueryIntent, EntityExtraction, QueryDecomposition
    )
    LLM_DECOMPOSER_AVAILABLE = True
    logger.info("LLM Query Decomposer loaded successfully")
except ImportError as e:
    LLM_DECOMPOSER_AVAILABLE = False
    logger.warning(f"LLM Query Decomposer not available: {e}")

# Import cache if available
try:
    from simple_cache import QueryCache
    query_cache = QueryCache(max_size=CACHE_MAX_SIZE, ttl_seconds=CACHE_TTL_SECONDS)
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False
    query_cache = None
    logger.warning("Cache not available")

# Initialize hybrid system
_hybrid_system = None

# Thread pool for parallel processing
executor = ThreadPoolExecutor(max_workers=3)

def get_hybrid_system() -> HybridProcurementRAG:
    """Singleton pattern for hybrid system"""
    global _hybrid_system
    if _hybrid_system is None:
        _hybrid_system = HybridProcurementRAG(enable_fuzzy_matching=True, fuzzy_threshold=0.8)
    return _hybrid_system

# ======================================================================
# ENHANCED QUERY PROCESSING WITH LLM
# ======================================================================

def answer_question_intelligent(question: str, mode: str = "auto") -> Dict[str, Any]:
    """
    Main entry point with LLM-powered understanding and comprehensive fallbacks
    
    Args:
        question: User's query
        mode: Processing mode - "auto", "sql", "rag", or "hybrid"
    
    Returns:
        Dict containing answer with natural language response and metadata
    """
    start_time = time.time()
    
    # Input sanitization
    question = sanitize_input(question)
    
    # Check cache first
    cache_key = None
    if CACHE_AVAILABLE and mode == "auto":
        cache_key = generate_cache_key(question)
        cached_result = query_cache.get(cache_key)
        if cached_result:
            logger.info(f"Cache hit for query: {question[:50]}...")
            cached_result['cache_hit'] = True
            return cached_result
    
    try:
        # ============================================
        # NEW: LLM-POWERED QUERY UNDERSTANDING
        # ============================================
        query_analysis = None
        if LLM_DECOMPOSER_AVAILABLE and mode in ["auto", "hybrid"]:
            logger.info("Analyzing query with LLM...")
            query_analysis = decompose_query(question)
            
            # Log the analysis
            logger.info(f"Query intent: {query_analysis['intent']['primary_intent']} "
                       f"(confidence: {query_analysis['intent']['confidence']})")
            logger.info(f"Extracted vendors: {query_analysis['entities']['vendors']}")
            logger.info(f"Query complexity: {'Complex' if query_analysis['is_complex'] else 'Simple'}")
        
        # Route based on mode and LLM analysis
        if mode == "sql":
            result = process_sql_query(question, query_analysis)
        elif mode == "rag":
            result = process_rag_query(question, query_analysis)
        elif mode == "hybrid":
            result = process_hybrid_query(question, query_analysis)
        else:  # auto mode
            result = process_auto_query_enhanced(question, query_analysis)
        
        # ============================================
        # NEW: NATURAL LANGUAGE RESPONSE GENERATION
        # ============================================
        if LLM_DECOMPOSER_AVAILABLE and result.get('answer'):
            # Generate natural language response
            original_answer = result.get('answer', '')
            enhanced_response = generate_response(question, result)
            
            # Keep both versions
            result['raw_answer'] = original_answer
            result['answer'] = enhanced_response
            result['llm_enhanced'] = True
        
        # Add metadata
        result['processing_time'] = time.time() - start_time
        result['mode'] = mode
        
        # Cache successful results
        if CACHE_AVAILABLE and cache_key and result.get('confidence', 0) > 70:
            query_cache.set(cache_key, result)
        
        return result
        
    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        return create_error_response(str(e), question)

def process_auto_query_enhanced(question: str, query_analysis: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Enhanced automatic query processing with LLM-powered routing
    """
    hybrid_system = get_hybrid_system()
    
    # ============================================
    # NEW: LLM-GUIDED PROCESSING
    # ============================================
    if query_analysis and LLM_DECOMPOSER_AVAILABLE:
        intent = query_analysis['intent']
        entities = query_analysis['entities']
        decomposition = query_analysis['decomposition']
        
        # Handle complex queries with decomposition
        if query_analysis['is_complex']:
            return process_complex_decomposed_query(question, decomposition, hybrid_system)
        
        # Route based on LLM-identified intent
        if intent['primary_intent'] == 'recommendation':
            # Needs both SQL data and semantic reasoning
            return process_recommendation_query(question, entities, hybrid_system)
        
        # For SQL-appropriate queries, enhance with extracted entities
        if intent['requires_sql'] and not intent['requires_semantic']:
            # Enhance question with resolved entities
            enhanced_question = enhance_question_with_llm_entities(question, entities)
            sql_result = try_sql_processing(enhanced_question, hybrid_system, entities)
            
            if sql_result and sql_result.get('confidence', 0) > 70:
                return sql_result
        
        # For semantic queries
        if intent['requires_semantic']:
            return process_semantic_enhanced(question, entities)
    
    # ============================================
    # FALLBACK TO ORIGINAL LOGIC
    # ============================================
    return process_auto_query_original(question, hybrid_system)

def process_complex_decomposed_query(question: str, decomposition: Dict, 
                                    hybrid_system: HybridProcurementRAG) -> Dict[str, Any]:
    """
    Process complex query using LLM decomposition
    """
    sub_queries = decomposition['sub_queries']
    execution_order = decomposition['execution_order']
    combination_strategy = decomposition['combination_strategy']
    
    results = {}
    combined_answer = []
    total_records = 0
    
    # Execute sub-queries in order
    for idx in execution_order:
        sub_query = sub_queries[idx]
        sub_question = sub_query['query']
        query_type = sub_query['type']
        dependencies = sub_query['dependencies']
        
        # Prepare context from dependencies
        context = {}
        for dep_idx in dependencies:
            if dep_idx in results:
                context[f'query_{dep_idx}'] = results[dep_idx]
        
        # Execute sub-query based on type
        if query_type == 'sql':
            sub_result = hybrid_system.process_query(sub_question)
        elif query_type == 'semantic':
            sub_result = process_rag_query(sub_question) if RAG_AVAILABLE else {}
        else:  # calculation
            sub_result = perform_calculation(sub_question, context)
        
        results[idx] = sub_result
        
        if sub_result.get('answer'):
            combined_answer.append(sub_result['answer'])
        
        if sub_result.get('records_analyzed'):
            total_records += sub_result['records_analyzed']
    
    # Combine results based on strategy
    if combination_strategy == 'merge':
        final_answer = '\n\n'.join(combined_answer)
    elif combination_strategy == 'sequential':
        final_answer = format_sequential_results(combined_answer)
    elif combination_strategy == 'conditional':
        final_answer = apply_conditional_logic(results, decomposition)
    else:
        final_answer = '\n\n'.join(combined_answer)
    
    return {
        'source': 'Complex Query Processing',
        'query_type': 'decomposed',
        'answer': final_answer,
        'sub_queries_executed': len(sub_queries),
        'records_analyzed': total_records,
        'confidence': 85,
        'decomposition': decomposition
    }

def process_recommendation_query(question: str, entities: Dict, 
                                hybrid_system: HybridProcurementRAG) -> Dict[str, Any]:
    """
    Process recommendation queries using both SQL and semantic reasoning
    """
    # Get performance data for relevant vendors
    vendor_data = {}
    
    if entities.get('vendors'):
        for vendor in entities['vendors']:
            stats = hybrid_system._get_vendor_statistics(vendor)
            if stats:
                vendor_data[vendor] = stats
    else:
        # Get all vendor statistics for general recommendations
        query = f"""
        SELECT 
            {hybrid_system.VENDOR_COL} as vendor,
            COUNT(*) as order_count,
            SUM(CAST({hybrid_system.COST_COL} AS FLOAT)) as total_spending,
            AVG(CAST({hybrid_system.COST_COL} AS FLOAT)) as avg_order,
            MIN(CAST({hybrid_system.COST_COL} AS FLOAT)) as min_order,
            MAX(CAST({hybrid_system.COST_COL} AS FLOAT)) as max_order
        FROM procurement
        WHERE {hybrid_system.COST_COL} IS NOT NULL
        GROUP BY {hybrid_system.VENDOR_COL}
        ORDER BY total_spending DESC
        LIMIT 20
        """
        
        df = pd.read_sql_query(query, hybrid_system.sql_conn)
        for _, row in df.iterrows():
            vendor_data[row['vendor']] = row.to_dict()
    
    # Apply business logic for recommendations
    recommendations = generate_business_recommendations(vendor_data, question, entities)
    
    return {
        'source': 'Recommendation Engine',
        'query_type': 'recommendation',
        'answer': recommendations,
        'vendors_analyzed': len(vendor_data),
        'confidence': 80,
        'evidence_report': 'Based on performance metrics and business logic analysis'
    }

def process_semantic_enhanced(question: str, entities: Dict) -> Dict[str, Any]:
    """
    Enhanced semantic search with entity awareness
    """
    if not RAG_AVAILABLE:
        return {
            'source': 'Semantic Search',
            'error': 'RAG module not available',
            'confidence': 0
        }
    
    # Enhance the question with extracted entities for better context
    enhanced_context = []
    
    if entities.get('vendors'):
        enhanced_context.append(f"Vendors of interest: {', '.join(entities['vendors'])}")
    
    if entities.get('metrics'):
        enhanced_context.append(f"Metrics to analyze: {', '.join(entities['metrics'])}")
    
    if entities.get('time_periods'):
        enhanced_context.append(f"Time period: {', '.join(entities['time_periods'])}")
    
    enhanced_question = question
    if enhanced_context:
        enhanced_question = f"{question}\n\nContext: {' | '.join(enhanced_context)}"
    
    # Call RAG with enhanced question
    rag_result = original_rag_answer(enhanced_question)
    rag_result['entities_used'] = entities
    
    return rag_result

# ======================================================================
# ENHANCED ENTITY RESOLUTION WITH LLM
# ======================================================================

def enhance_question_with_llm_entities(question: str, entities: Dict) -> str:
    """
    Enhance question with LLM-extracted and resolved entities
    """
    enhanced = question
    
    # Handle ambiguous references
    if entities.get('ambiguous_references'):
        for ambiguous, resolved in entities['ambiguous_references'].items():
            logger.info(f"Resolving '{ambiguous}' to '{resolved}'")
            # Replace in question for SQL processing
            enhanced = enhanced.replace(ambiguous, resolved)
    
    # Add vendor clarifications if needed
    if entities.get('vendors') and not any(v.lower() in question.lower() for v in entities['vendors']):
        vendors_str = ' and '.join(entities['vendors'][:3])
        enhanced = f"{enhanced} (vendors: {vendors_str})"
    
    return enhanced

def extract_vendors_with_llm(question: str, hybrid_system: HybridProcurementRAG) -> List[str]:
    """
    Extract vendors using LLM for better understanding
    """
    if not LLM_DECOMPOSER_AVAILABLE:
        return extract_vendors_with_fallbacks_original(question, hybrid_system)
    
    # Use LLM to resolve ambiguous references
    decomposer = get_decomposer()
    entities = decomposer.extract_entities(question)
    
    vendors = entities.vendors
    
    # Also check for ambiguous references
    if entities.ambiguous_references:
        for ref, possible_vendor in entities.ambiguous_references.items():
            # Resolve each reference
            resolved = resolve_reference(ref, context="procurement vendors")
            vendors.extend(resolved)
    
    # Validate against database
    validated_vendors = []
    for vendor in vendors:
        db_vendors = hybrid_system._find_vendor_in_db(vendor)
        validated_vendors.extend(db_vendors)
    
    return list(set(validated_vendors))[:10]

# ======================================================================
# ENHANCED SQL PROCESSING WITH LLM UNDERSTANDING
# ======================================================================

def try_sql_processing(question: str, hybrid_system: HybridProcurementRAG, 
                      entities: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Enhanced SQL processing with LLM entity understanding
    """
    try:
        # If we have LLM-extracted entities, use them
        if entities and entities.get('vendors'):
            # Modify the hybrid system's vendor extraction to use our entities
            original_extract = hybrid_system._extract_vendor_names
            hybrid_system._extract_vendor_names = lambda q: entities['vendors']
        
        # Let hybrid system handle the query
        result = hybrid_system.process_query(question)
        
        # Restore original method if modified
        if entities and entities.get('vendors'):
            hybrid_system._extract_vendor_names = original_extract
        
        # If SQL succeeded, enhance the result
        if result.get('source') == 'SQL' and result.get('answer'):
            # Add entity information to result
            if entities:
                result['extracted_entities'] = entities
            return result
        
        # Statistical Query Enhancement with LLM understanding
        if entities and 'median' in entities.get('metrics', []):
            return calculate_statistics_enhanced(question, entities, hybrid_system)
        
        return result
        
    except Exception as e:
        logger.error(f"SQL processing failed: {e}")
        return {"error": str(e), "source": "SQL", "confidence": 0}

def calculate_statistics_enhanced(question: str, entities: Dict, 
                                 hybrid_system: HybridProcurementRAG) -> Dict[str, Any]:
    """
    Enhanced statistical calculation with entity awareness
    """
    try:
        # Build query with entity filters
        base_query = f"""
        SELECT CAST({hybrid_system.COST_COL} AS FLOAT) as value
        FROM procurement
        WHERE {hybrid_system.COST_COL} IS NOT NULL
        AND CAST({hybrid_system.COST_COL} AS FLOAT) > 0
        """
        
        # Add vendor filters if specified
        if entities.get('vendors'):
            vendor_conditions = []
            for vendor in entities['vendors']:
                vendor_conditions.append(f"UPPER({hybrid_system.VENDOR_COL}) LIKE '%{vendor.upper()}%'")
            if vendor_conditions:
                base_query += f" AND ({' OR '.join(vendor_conditions)})"
        
        df = pd.read_sql_query(base_query, hybrid_system.sql_conn)
        
        if df.empty:
            return {
                "answer": "No data available for statistical calculation with specified filters",
                "source": "SQL",
                "confidence": 0
            }
        
        values = df['value'].values
        
        # Calculate all requested metrics
        results = {}
        for metric in entities.get('metrics', []):
            metric_lower = metric.lower()
            
            if 'median' in metric_lower:
                results['median'] = np.median(values)
            elif 'mean' in metric_lower or 'average' in metric_lower:
                results['mean'] = np.mean(values)
            elif 'variance' in metric_lower:
                results['variance'] = np.var(values)
            elif 'standard deviation' in metric_lower or 'stddev' in metric_lower:
                results['std_dev'] = np.std(values)
            elif 'min' in metric_lower:
                results['min'] = np.min(values)
            elif 'max' in metric_lower:
                results['max'] = np.max(values)
        
        # Format comprehensive answer
        answer_parts = [f"Statistical Analysis ({len(values):,} records):"]
        for metric, value in results.items():
            answer_parts.append(f"- {metric.replace('_', ' ').title()}: ${value:,.2f}")
        
        if entities.get('vendors'):
            answer_parts.append(f"\nFiltered for vendors: {', '.join(entities['vendors'])}")
        
        return {
            "answer": '\n'.join(answer_parts),
            "source": "SQL",
            "query_type": "statistical",
            "records_analyzed": len(values),
            "statistics": results,
            "confidence": 95
        }
        
    except Exception as e:
        logger.error(f"Enhanced statistical calculation failed: {e}")
        return {"error": str(e), "source": "SQL", "confidence": 0}

# ======================================================================
# BUSINESS LOGIC AND RECOMMENDATIONS
# ======================================================================

def generate_business_recommendations(vendor_data: Dict, question: str, entities: Dict) -> str:
    """
    Generate business recommendations based on data analysis
    """
    recommendations = []
    
    # Analyze vendor performance
    if vendor_data:
        # Calculate metrics
        avg_spending = np.mean([v.get('total_spending', 0) for v in vendor_data.values()])
        avg_orders = np.mean([v.get('order_count', 0) for v in vendor_data.values()])
        
        # Identify underperformers
        underperformers = []
        overperformers = []
        
        for vendor, stats in vendor_data.items():
            spending = stats.get('total_spending', 0)
            orders = stats.get('order_count', 0)
            avg_order = stats.get('avg_order', 0)
            
            # Simple performance scoring
            if orders < avg_orders * 0.5 and spending < avg_spending * 0.5:
                underperformers.append({
                    'vendor': vendor,
                    'reason': f"Low activity: {orders} orders, ${spending:,.2f} total"
                })
            elif spending > avg_spending * 2 and orders > avg_orders * 1.5:
                overperformers.append({
                    'vendor': vendor,
                    'reason': f"High volume: {orders} orders, ${spending:,.2f} total"
                })
        
        # Generate recommendations
        if 'drop' in question.lower() or 'remove' in question.lower():
            recommendations.append("### Vendor Consolidation Recommendations\n")
            if underperformers:
                recommendations.append("**Consider reviewing relationships with:**")
                for vendor in underperformers[:5]:
                    recommendations.append(f"- **{vendor['vendor']}**: {vendor['reason']}")
                recommendations.append("\n*Rationale*: These vendors show significantly below-average activity.")
            else:
                recommendations.append("All vendors show reasonable activity levels.")
        
        elif 'invest' in question.lower() or 'increase' in question.lower():
            recommendations.append("### Strategic Partnership Recommendations\n")
            if overperformers:
                recommendations.append("**Consider strengthening relationships with:**")
                for vendor in overperformers[:5]:
                    recommendations.append(f"- **{vendor['vendor']}**: {vendor['reason']}")
                recommendations.append("\n*Rationale*: These vendors demonstrate strong engagement.")
        
        else:
            # General recommendations
            recommendations.append("### Procurement Analysis Summary\n")
            recommendations.append(f"- **Total Vendors Analyzed**: {len(vendor_data)}")
            recommendations.append(f"- **Average Spending per Vendor**: ${avg_spending:,.2f}")
            recommendations.append(f"- **Average Orders per Vendor**: {avg_orders:.0f}")
            
            if underperformers:
                recommendations.append(f"\n**Optimization Opportunity**: {len(underperformers)} vendors "
                                     "show low activity and could be candidates for consolidation.")
            
            if overperformers:
                recommendations.append(f"\n**Key Partners**: {len(overperformers)} vendors "
                                     "represent your primary procurement relationships.")
    
    if not recommendations:
        recommendations.append("Unable to generate specific recommendations. "
                             "Please provide more context or specific criteria.")
    
    return '\n'.join(recommendations)

# ======================================================================
# PARALLEL PROCESSING FOR COMPLEX QUERIES
# ======================================================================

def process_parallel_subqueries(sub_queries: List[Dict], hybrid_system: HybridProcurementRAG) -> Dict[int, Any]:
    """
    Process independent sub-queries in parallel
    """
    results = {}
    
    # Identify independent queries (no dependencies)
    independent = [(i, sq) for i, sq in enumerate(sub_queries) if not sq.get('dependencies', [])]
    
    if independent:
        futures = {}
        for idx, sub_query in independent:
            future = executor.submit(
                hybrid_system.process_query,
                sub_query['query']
            )
            futures[future] = idx
        
        # Collect results
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result(timeout=10)
            except Exception as e:
                logger.error(f"Sub-query {idx} failed: {e}")
                results[idx] = {"error": str(e)}
    
    return results

# ======================================================================
# HELPER FUNCTIONS
# ======================================================================

def perform_calculation(query: str, context: Dict) -> Dict[str, Any]:
    """
    Perform calculations on previous query results
    """
    try:
        # Extract values from context
        values = []
        for key, result in context.items():
            if isinstance(result, dict):
                if 'total_spending' in result:
                    values.append(result['total_spending'])
                elif 'avg_order' in result:
                    values.append(result['avg_order'])
        
        if not values:
            return {"error": "No values to calculate", "confidence": 0}
        
        # Perform calculation based on query
        query_lower = query.lower()
        
        if 'compare' in query_lower or 'better' in query_lower:
            if len(values) >= 2:
                diff = values[0] - values[1]
                pct_diff = (diff / values[1]) * 100 if values[1] != 0 else 0
                
                answer = f"Comparison Result:\n"
                answer += f"- First value: ${values[0]:,.2f}\n"
                answer += f"- Second value: ${values[1]:,.2f}\n"
                answer += f"- Difference: ${abs(diff):,.2f} ({abs(pct_diff):.1f}%)\n"
                answer += f"- {'First' if diff > 0 else 'Second'} is higher"
                
                return {
                    "answer": answer,
                    "source": "Calculation",
                    "confidence": 100
                }
        
        return {
            "answer": f"Calculated values: {values}",
            "source": "Calculation",
            "confidence": 80
        }
        
    except Exception as e:
        logger.error(f"Calculation failed: {e}")
        return {"error": str(e), "confidence": 0}

def format_sequential_results(results: List[str]) -> str:
    """Format sequential results with step numbers"""
    formatted = []
    for i, result in enumerate(results, 1):
        formatted.append(f"**Step {i}:**\n{result}")
    return '\n\n'.join(formatted)

def apply_conditional_logic(results: Dict, decomposition: Dict) -> str:
    """Apply conditional logic to combine results"""
    # This would need more complex implementation based on specific conditions
    # For now, return all results
    combined = []
    for idx, result in results.items():
        if result.get('answer'):
            combined.append(result['answer'])
    return '\n\n'.join(combined)

# ======================================================================
# ORIGINAL FALLBACK FUNCTIONS (Preserved)
# ======================================================================

def process_auto_query_original(question: str, hybrid_system: HybridProcurementRAG) -> Dict[str, Any]:
    """Original auto query processing as fallback"""
    try:
        # First attempt - SQL processing
        sql_result = try_sql_processing_original(question, hybrid_system)
        sql_confidence = calculate_confidence(sql_result, "sql")
        
        # If SQL has high confidence, return it
        if sql_confidence > 80:
            logger.info(f"SQL result with high confidence ({sql_confidence})")
            return sql_result
        
        # Second attempt - RAG processing
        if RAG_AVAILABLE:
            rag_result = try_rag_processing(question)
            rag_confidence = calculate_confidence(rag_result, "rag")
            
            # Merge if both have results
            if sql_confidence > 50 and rag_confidence > 50:
                logger.info("Merging SQL and RAG results")
                return merge_results_enhanced(sql_result, rag_result, question)
            
            # Return best result
            if rag_confidence > sql_confidence:
                return rag_result
        
        # If we have any SQL result, return it
        if sql_result and sql_result.get('answer'):
            return sql_result
        
        # Final fallback - basic keyword search
        return perform_basic_search(question, hybrid_system)
        
    except Exception as e:
        logger.error(f"Auto query processing failed: {e}")
        return create_error_response(str(e), question)

def try_sql_processing_original(question: str, hybrid_system: HybridProcurementRAG) -> Dict[str, Any]:
    """Original SQL processing without LLM enhancement"""
    try:
        result = hybrid_system.process_query(question)
        
        if result.get('source') == 'SQL' and result.get('answer'):
            return result
        
        # Try vendor resolution fallback
        vendors = extract_vendors_with_fallbacks_original(question, hybrid_system)
        if vendors:
            modified_question = enhance_question_with_vendors(question, vendors)
            result = hybrid_system.process_query(modified_question)
            if result.get('answer'):
                return result
        
        # Statistical query fallback
        if is_statistical_query(question):
            return calculate_statistics_manually_original(question, hybrid_system)
        
        return result
        
    except Exception as e:
        logger.error(f"SQL processing failed: {e}")
        return {"error": str(e), "source": "SQL", "confidence": 0}

def extract_vendors_with_fallbacks_original(question: str, hybrid_system: HybridProcurementRAG) -> List[str]:
    """Original vendor extraction without LLM"""
    vendors = []
    
    # Strategy 1: Direct extraction from hybrid system
    vendors = hybrid_system._extract_vendor_names(question)
    if vendors:
        return vendors
    
    # Strategy 2: Fuzzy matching
    words = question.upper().split()
    for word in words:
        if len(word) > 3:
            fuzzy_matches = hybrid_system._find_fuzzy_vendor_matches(word)
            vendors.extend(fuzzy_matches)
    
    if vendors:
        return list(set(vendors))[:10]
    
    # Strategy 3: Pattern matching
    patterns = [
        r'\b([A-Z][A-Z]+)\b',  # All caps words
        r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',  # Proper nouns
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, question)
        for match in matches:
            if len(match) > 2:
                db_vendors = hybrid_system._find_vendor_in_db(match)
                vendors.extend(db_vendors)
    
    return list(set(vendors))[:10]

def calculate_statistics_manually_original(question: str, hybrid_system: HybridProcurementRAG) -> Dict[str, Any]:
    """Original statistical calculation without LLM enhancement"""
    try:
        query = f"""
        SELECT CAST({hybrid_system.COST_COL} AS FLOAT) as value
        FROM procurement
        WHERE {hybrid_system.COST_COL} IS NOT NULL
        AND CAST({hybrid_system.COST_COL} AS FLOAT) > 0
        """
        
        df = pd.read_sql_query(query, hybrid_system.sql_conn)
        
        if df.empty:
            return {
                "answer": "No data available for statistical calculation",
                "source": "SQL",
                "confidence": 0
            }
        
        values = df['value'].values
        question_lower = question.lower()
        
        # Calculate requested statistic
        result_text = ""
        
        if 'median' in question_lower:
            result = np.median(values)
            result_text = f"Median: ${result:,.2f}"
        
        if 'mean' in question_lower or 'average' in question_lower:
            result = np.mean(values)
            result_text += f"\nMean: ${result:,.2f}"
        
        if not result_text:
            result_text = f"""Statistical Summary:
- Count: {len(values):,}
- Mean: ${np.mean(values):,.2f}
- Median: ${np.median(values):,.2f}
- Std Dev: ${np.std(values):,.2f}"""
        
        return {
            "answer": result_text,
            "source": "SQL",
            "query_type": "statistical",
            "records_analyzed": len(values),
            "confidence": 95
        }
        
    except Exception as e:
        logger.error(f"Statistical calculation failed: {e}")
        return {"error": str(e), "source": "SQL", "confidence": 0}

# ======================================================================
# ENHANCED RESULT MERGING WITH LLM
# ======================================================================

def merge_results_enhanced(sql_result: Dict[str, Any], rag_result: Dict[str, Any], 
                          question: str) -> Dict[str, Any]:
    """
    Enhanced result merging with LLM-powered synthesis
    """
    sql_conf = sql_result.get('confidence', 0)
    rag_conf = rag_result.get('confidence', 0)
    
    # If one is significantly better, use it
    if sql_conf > rag_conf + 20:
        return sql_result
    if rag_conf > sql_conf + 20:
        return rag_result
    
    # Merge the results
    merged = {
        "source": "Hybrid",
        "query_type": "merged",
        "confidence": max(sql_conf, rag_conf),
        "sql_confidence": sql_conf,
        "rag_confidence": rag_conf
    }
    
    # If LLM is available, synthesize the results
    if LLM_DECOMPOSER_AVAILABLE:
        combined_data = {
            'sql_result': sql_result.get('answer', ''),
            'rag_result': rag_result.get('answer', ''),
            'sql_records': sql_result.get('records_analyzed', 0),
            'rag_records': rag_result.get('records_analyzed', 0)
        }
        
        synthesized = generate_response(question, combined_data)
        merged['answer'] = synthesized
        merged['synthesis_method'] = 'llm'
    else:
        # Fallback to simple combination
        if sql_conf > 70 and rag_conf > 70:
            merged['answer'] = f"""**Structured Data Analysis:**
{sql_result.get('answer', 'No SQL results')}

**Document Analysis:**
{rag_result.get('answer', 'No RAG results')}"""
        elif sql_conf > 70:
            merged['answer'] = sql_result.get('answer')
            merged['additional_context'] = rag_result.get('answer')
        else:
            merged['answer'] = rag_result.get('answer')
            merged['additional_context'] = sql_result.get('answer')
    
    # Merge metadata
    merged['records_analyzed'] = (
        sql_result.get('records_analyzed', 0) + 
        rag_result.get('records_analyzed', 0)
    )
    
    return merged

# ======================================================================
# REMAINING HELPER FUNCTIONS (Preserved from original)
# ======================================================================

def process_sql_query(question: str, query_analysis: Optional[Dict] = None) -> Dict[str, Any]:
    """Force SQL processing"""
    hybrid_system = get_hybrid_system()
    
    if query_analysis and LLM_DECOMPOSER_AVAILABLE:
        entities = query_analysis.get('entities', {})
        return try_sql_processing(question, hybrid_system, entities)
    
    return try_sql_processing_original(question, hybrid_system)

def process_rag_query(question: str, query_analysis: Optional[Dict] = None) -> Dict[str, Any]:
    """Force RAG processing"""
    if query_analysis and LLM_DECOMPOSER_AVAILABLE:
        entities = query_analysis.get('entities', {})
        return process_semantic_enhanced(question, entities)
    
    return try_rag_processing(question)

def process_hybrid_query(question: str, query_analysis: Optional[Dict] = None) -> Dict[str, Any]:
    """Process with both SQL and RAG, then merge"""
    hybrid_system = get_hybrid_system()
    
    sql_result = process_sql_query(question, query_analysis)
    rag_result = process_rag_query(question, query_analysis) if RAG_AVAILABLE else None
    
    if rag_result and sql_result:
        return merge_results_enhanced(sql_result, rag_result, question)
    
    return sql_result or rag_result or create_error_response("No results", question)

def try_rag_processing(question: str) -> Dict[str, Any]:
    """Attempt RAG processing with fallbacks"""
    if not RAG_AVAILABLE:
        return {"error": "RAG not available", "source": "RAG", "confidence": 0}
    
    try:
        rag_result = original_rag_answer(question)
        
        if rag_result and not rag_result.get('confidence'):
            rag_result['confidence'] = calculate_confidence(rag_result, "rag")
        
        return rag_result
        
    except Exception as e:
        logger.error(f"RAG processing failed: {e}")
        return {"error": str(e), "source": "RAG", "confidence": 0}

def calculate_confidence(result: Dict[str, Any], source_type: str) -> float:
    """Calculate confidence score for a result"""
    confidence = 50.0
    
    if not result or 'error' in result:
        return 0.0
    
    if source_type == "sql":
        if result.get('source') == 'SQL':
            confidence += 20
        if result.get('records_analyzed', 0) > 0:
            confidence += min(30, result['records_analyzed'] / 10)
        if result.get('query_type') in ['comparison', 'aggregation', 'ranking']:
            confidence += 20
    
    elif source_type == "rag":
        if result.get('answer') and len(str(result.get('answer', ''))) > 100:
            confidence += 20
        generic_phrases = ['cannot be calculated', 'no data available', 
                          'unable to determine', 'not enough information']
        answer_lower = str(result.get('answer', '')).lower()
        if any(phrase in answer_lower for phrase in generic_phrases):
            confidence -= 30
    
    return min(100.0, max(0.0, confidence))

def is_statistical_query(question: str) -> bool:
    """Check if query requires statistical calculation"""
    question_lower = question.lower()
    return any(term in question_lower for term in STATISTICAL_KEYWORDS)

def enhance_question_with_vendors(question: str, vendors: List[str]) -> str:
    """Enhance question by explicitly including vendor names"""
    vendor_str = " and ".join(vendors[:3])
    
    if not any(v.lower() in question.lower() for v in vendors):
        if "compare" in question.lower():
            return f"{question} (vendors: {vendor_str})"
        else:
            return f"{question} for {vendor_str}"
    
    return question

def perform_basic_search(question: str, hybrid_system: HybridProcurementRAG) -> Dict[str, Any]:
    """Basic keyword search as final fallback"""
    try:
        keywords = extract_keywords(question)
        
        if not keywords:
            return create_error_response("No searchable keywords found", question)
        
        conditions = []
        params = []
        
        for keyword in keywords[:5]:
            conditions.append(f"""
                (UPPER({hybrid_system.VENDOR_COL}) LIKE ? OR 
                 UPPER({hybrid_system.DESC_COL}) LIKE ? OR
                 UPPER({hybrid_system.COMMODITY_COL}) LIKE ?)
            """)
            pattern = f"%{keyword.upper()}%"
            params.extend([pattern, pattern, pattern])
        
        query = f"""
        SELECT 
            {hybrid_system.VENDOR_COL} as vendor,
            {hybrid_system.DESC_COL} as description,
            {hybrid_system.COMMODITY_COL} as commodity,
            {hybrid_system.COST_COL} as cost
        FROM procurement
        WHERE {' OR '.join(conditions)}
        LIMIT 20
        """
        
        df = pd.read_sql_query(query, hybrid_system.sql_conn, params=params)
        
        if df.empty:
            return {
                "answer": "No results found for your query. Please try different keywords.",
                "source": "Basic Search",
                "confidence": 10
            }
        
        answer = f"Found {len(df)} results matching keywords: {', '.join(keywords)}\n\n"
        
        for i, row in enumerate(df.head(5).itertuples(), 1):
            answer += f"{i}. Vendor: {row.vendor}\n"
            answer += f"   Description: {row.description[:100]}...\n"
            answer += f"   Cost: ${float(row.cost):,.2f}\n\n"
        
        if len(df) > 5:
            answer += f"... and {len(df) - 5} more results"
        
        return {
            "answer": answer,
            "source": "Basic Search",
            "query_type": "keyword_search",
            "records_analyzed": len(df),
            "confidence": 40
        }
        
    except Exception as e:
        logger.error(f"Basic search failed: {e}")
        return create_error_response(str(e), question)

def extract_keywords(question: str) -> List[str]:
    """Extract meaningful keywords from question"""
    words = question.lower().split()
    keywords = []
    
    for word in words:
        word = re.sub(r'[^\w]', '', word)
        
        if len(word) > 2 and word not in STOP_WORDS:
            keywords.append(word)
    
    return keywords

def sanitize_input(question: str) -> str:
    """Sanitize user input to prevent SQL injection"""
    if not question:
        return ""
    
    sanitized = question
    for keyword in SQL_INJECTION_KEYWORDS:
        if f' {keyword} ' in sanitized.upper() or sanitized.upper().startswith(keyword):
            sanitized = re.sub(f'\\b{keyword}\\b', '', sanitized, flags=re.IGNORECASE)
    
    sanitized = ' '.join(sanitized.split())
    
    try:
        sanitized = sanitized.encode('utf-8', errors='ignore').decode('utf-8')
    except:
        pass
    
    return sanitized.strip()

def generate_cache_key(question: str) -> str:
    """Generate cache key for question"""
    normalized = question.lower().strip()
    normalized = ' '.join(normalized.split())
    return hashlib.md5(normalized.encode()).hexdigest()

def create_error_response(error_msg: str, question: str) -> Dict[str, Any]:
    """Create structured error response"""
    error_context = {
        "error": error_msg,
        "source": "Error Handler",
        "query_type": "error",
        "confidence": 0,
        "timestamp": datetime.now().isoformat(),
        "original_question": question[:100]
    }
    
    error_lower = error_msg.lower()
    
    if "timeout" in error_lower:
        error_context["suggestion"] = "Query took too long. Try simplifying your question."
    elif "vendor" in error_lower:
        error_context["suggestion"] = "Vendor not found. Try using the full vendor name or check spelling."
    elif "no data" in error_lower or "no results" in error_lower:
        error_context["suggestion"] = "No matching data found. Try broader search terms."
    elif "sql" in error_lower or "database" in error_lower:
        error_context["suggestion"] = "Database error. Please try again or contact support."
    else:
        error_context["suggestion"] = "An error occurred. Please rephrase your question and try again."
    
    error_context["answer"] = f"""
I encountered an issue processing your query: {error_context['suggestion']}

If you continue to experience issues, you can try:
1. Simplifying your question
2. Using specific vendor names
3. Asking for basic statistics first
"""
    
    return error_context

# ======================================================================
# TESTING ENHANCED SYSTEM
# ======================================================================

def test_enhanced_system():
    """Test the enhanced LLM-powered system"""
    test_queries = [
        "Compare Dell and IBM and tell me which one we should invest more in",
        "What's the median order value for that big computer company?",
        "Which vendors should we consider dropping based on poor performance?",
        "Show spending trends and recommend cost optimization strategies",
        "How much did we spend with Microsft last quarter?",  # Typo intentional
        "'; DROP TABLE procurement; --",  # SQL injection test
    ]
    
    print("Testing Enhanced LLM-Powered System:")
    print("=" * 60)
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")
        try:
            result = answer_question_intelligent(query)
            
            print(f"Source: {result.get('source')}")
            print(f"Query Type: {result.get('query_type')}")
            print(f"Confidence: {result.get('confidence')}")
            print(f"LLM Enhanced: {result.get('llm_enhanced', False)}")
            
            if result.get('extracted_entities'):
                print(f"Entities: {result['extracted_entities'].get('vendors', [])}")
            
            if result.get('decomposition'):
                print(f"Complex Query: {result['decomposition']['is_complex']}")
                print(f"Sub-queries: {result['decomposition'].get('sub_queries_executed', 0)}")
            
            print(f"\nAnswer Preview:")
            print(result.get('answer', 'No answer')[:300] + "...")
            
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    test_enhanced_system()
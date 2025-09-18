"""
app.py - Production Flask Application with All Optimizations
Integrates: Smart Routing, Unified Analysis, VendorResolver, Grounded Prompts, Enhanced Caching, Template Parsing
UPDATED: Simplified template handling without non-critical fallbacks
"""

import os
import sys
import json
import hashlib
import time
from datetime import datetime
import pandas as pd
import numpy as np
from contextlib import contextmanager
import logging
from typing import Dict, Any, Optional, List, Tuple
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Import all helper functions (now with template support)
from app_helpers import (
    # Template extraction
    extract_text_from_response,
    extract_from_template_response,
    # Dashboard functions
    get_dashboard_summary, get_trend_data, generate_alerts,
    generate_dashboard_recommendations, generate_executive_summary,
    # Report functions
    generate_report_section, generate_report_conclusions,
    generate_report_recommendations, generate_report_visualizations,
    # Insight functions (now grounded)
    generate_vendor_insights, generate_spending_insights,
    generate_efficiency_insights,
    # Vendor functions (now using VendorResolver)
    get_vendor_comprehensive_data, generate_vendor_analysis,
    assess_vendor_risk, identify_vendor_opportunities,
    generate_vendor_insights_from_data, 
    generate_vendor_recommendations as generate_vendor_recs_from_data,
    # Analysis functions
    analyze_vendor_comprehensive, analyze_vendor_sql,
    analyze_spending_patterns, analyze_optimization_opportunities,
    perform_general_analysis,
    # Comparison and stats
    compare_vendors_sql, generate_comparison_visualization,
    calculate_statistical_metrics, interpret_statistics,
    assess_statistical_significance, suggest_visualization,
    # Other utilities
    process_conversational_query, generate_sql_recommendations,
    generate_action_items, generate_priority_matrix,
    combine_analysis_results, identify_spending_patterns,
    check_data_sufficiency
)

from constants import (
    DB_PATH, VENDOR_COL, COST_COL, DESC_COL, COMMODITY_COL,
    DEFAULT_PORT, DEFAULT_HOST, LOG_LEVEL, LOG_FORMAT,
    CACHE_MAX_SIZE, CACHE_TTL_SECONDS,
    # Import feature flags and performance settings
    FEATURES, PERFORMANCE_TARGETS, SLOW_QUERY_THRESHOLD,
    MIN_DATA_REQUIREMENTS, INSUFFICIENT_DATA_MESSAGES
)
from database_utils import db_manager, safe_execute_query

from dotenv import load_dotenv
load_dotenv()

# Import main query processor (with all optimizations)
from hybrid_rag_logic import (
    answer_question_intelligent, 
    sanitize_input, 
    generate_cache_key,
    get_system_performance_stats
)
from hybrid_rag_architecture import HybridProcurementRAG, get_vendor_resolver

# Configure logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# Check for LLM components
try:
    from query_decomposer import (
        get_decomposer, decompose_query, generate_response, 
        resolve_reference, get_performance_stats as get_llm_stats
    )
    LLM_DECOMPOSER_AVAILABLE = True
except ImportError as e:
    LLM_DECOMPOSER_AVAILABLE = False
    logger.warning(f"LLM Query Decomposer not available: {e}")

# Check for enhanced RAG
from app_helpers import get_strategic_recommendations

try:
    from rag_logic import (
        get_rag_processor, analyze_vendor_semantic, 
        compare_vendors_semantic
    )
    ENHANCED_RAG_AVAILABLE = True
except ImportError as e:
    ENHANCED_RAG_AVAILABLE = False
    logger.warning(f"Enhanced RAG Processor not available: {e}")

# Import enhanced cache manager
try:
    from simple_cache import (
        get_cache_manager, get_cache_stats, 
        optimize_caches, MultiCacheManager
    )
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False
    get_cache_manager = None
    logger.warning("Enhanced cache not available")

# Flask imports
from flask import Flask, request, jsonify, Response, render_template_string
try:
    from flask_cors import CORS
    CORS_AVAILABLE = True
except ImportError:
    CORS_AVAILABLE = False
    logger.warning("Flask-CORS not installed. CORS support disabled.")

# Custom JSON encoder for numpy types
class NumpyJSONProvider(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray): 
            return obj.tolist()
        elif isinstance(obj, pd.Series): 
            return obj.tolist()
        elif pd.isna(obj): 
            return None
        return super().default(obj)

# Initialize Flask app
app = Flask(__name__)
app.json_encoder = NumpyJSONProvider

if CORS_AVAILABLE:
    CORS(app)
    logger.info("CORS enabled for cross-origin requests")

# Initialize hybrid system (singleton)
_hybrid_system = None
def get_hybrid_system() -> HybridProcurementRAG:
    global _hybrid_system
    if _hybrid_system is None:
        _hybrid_system = HybridProcurementRAG(
            enable_fuzzy_matching=True,
            fuzzy_threshold=0.8,
            use_llm=LLM_DECOMPOSER_AVAILABLE
        )
    return _hybrid_system

# Initialize LLM components
_decomposer, _rag_processor = None, None
def get_llm_components():
    global _decomposer, _rag_processor
    if LLM_DECOMPOSER_AVAILABLE and _decomposer is None: 
        _decomposer = get_decomposer()
    if ENHANCED_RAG_AVAILABLE and _rag_processor is None: 
        _rag_processor = get_rag_processor()
    return _decomposer, _rag_processor

# ============================================
# PERFORMANCE MONITORING MIDDLEWARE
# ============================================

@app.before_request
def before_request():
    """Track request start time"""
    request.start_time = time.time()

@app.after_request
def after_request(response):
    """Log performance metrics"""
    if hasattr(request, 'start_time'):
        elapsed = time.time() - request.start_time
        
        # Log slow requests
        if elapsed > SLOW_QUERY_THRESHOLD:
            logger.warning(f"Slow request: {request.path} took {elapsed:.2f}s")
        
        # Add performance header
        response.headers['X-Processing-Time'] = f"{elapsed:.3f}"
    
    return response

# ============================================
# MAIN QUERY ENDPOINTS (WITH ALL OPTIMIZATIONS)
# ============================================

@app.route('/ask', methods=['POST'])
def ask():
    """
    Main query endpoint with all optimizations including template parsing.
    """
    try:
        data = request.json
        if not data or 'question' not in data:
            return jsonify({"error": "Missing 'question' in request"}), 400
        
        question = data.get('question', '')
        mode = data.get('mode', 'auto')
        
        # Performance tracking
        start_time = time.time()
        
        # Process with all optimizations
        result = answer_question_intelligent(question, mode=mode)
        
        # Extract template content if present (direct extraction, no fallback)
        if 'answer' in result:
            result['answer'] = extract_text_from_response(result['answer'])
        
        # Add metadata
        result['endpoint'] = '/ask'
        result['mode_requested'] = mode
        result['optimizations_active'] = {
            'smart_routing': FEATURES.get('smart_routing', False),
            'unified_analysis': FEATURES.get('unified_analysis', False),
            'vendor_resolver': FEATURES.get('central_vendor_resolver', False),
            'grounded_prompts': FEATURES.get('grounded_prompts', False),
            'granular_caching': FEATURES.get('granular_caching', False),
            'template_parsing': FEATURES.get('template_parsing', False)
        }
        
        # Log performance
        elapsed = time.time() - start_time
        if elapsed < PERFORMANCE_TARGETS.get('simple_query', 2.0) and result.get('llm_bypassed'):
            logger.info(f"✅ Fast response via smart routing: {elapsed:.2f}s")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in /ask endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/ask-advanced', methods=['POST'])
def ask_advanced():
    """
    Advanced query endpoint with detailed analysis and template parsing visibility.
    """
    if not LLM_DECOMPOSER_AVAILABLE:
        return jsonify({"error": "Advanced features require LLM components"}), 503
    
    try:
        data = request.json
        question = data.get('question')
        if not question:
            return jsonify({"error": "Missing 'question' in request"}), 400
        
        # Get detailed query analysis (uses unified analysis if enabled)
        query_analysis = decompose_query(question)
        
        # Process with hybrid mode
        result = answer_question_intelligent(question, mode='hybrid')
        
        # Extract template content
        if 'answer' in result:
            original_answer = result['answer']
            result['answer'] = extract_text_from_response(original_answer)
            result['template_extracted'] = (original_answer != result['answer'])
        
        # Add detailed analysis
        result['query_analysis'] = query_analysis
        result['optimization_path'] = []
        
        # Document optimization path
        if result.get('llm_bypassed'):
            result['optimization_path'].append('Smart Routing → Direct SQL')
        elif query_analysis.get('cache_hit'):
            result['optimization_path'].append('Cache Hit → Instant Response')
        else:
            if FEATURES.get('unified_analysis'):
                result['optimization_path'].append('Unified Analysis (1 LLM call)')
            else:
                result['optimization_path'].append('Traditional Analysis (3-4 LLM calls)')
            
            if FEATURES.get('template_parsing'):
                result['optimization_path'].append('Template-based Response Generation')
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in /ask-advanced endpoint: {e}")
        return jsonify({"error": str(e)}), 500

# ============================================
# VENDOR ENDPOINTS (WITH VENDORRESOLVER)
# ============================================

@app.route('/vendor/<vendor_name>', methods=['GET'])
def vendor_details(vendor_name):
    """
    Get vendor details with VendorResolver fuzzy matching.
    """
    try:
        # Use comprehensive analysis (includes VendorResolver)
        data = analyze_vendor_comprehensive(vendor_name)
        
        if "error" in data:
            # Check if suggestions available
            if "suggestions" in data:
                return jsonify({
                    "error": data["error"],
                    "suggestions": data["suggestions"],
                    "message": data.get("message", "Vendor not found")
                }), 404
            return jsonify(data), 404
        
        # Extract template content from analysis if present
        if 'analysis' in data:
            data['analysis'] = extract_text_from_response(data['analysis'])
        
        # Add resolver status
        data['vendor_resolution'] = {
            'input': vendor_name,
            'resolved_to': data['vendor_data'].get('vendor'),
            'resolver_enabled': FEATURES.get('central_vendor_resolver', False),
            'template_parsing': FEATURES.get('template_parsing', False)
        }
        
        return jsonify(data)
        
    except Exception as e:
        logger.error(f"Error getting vendor details: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/resolve-vendor/<vendor_input>', methods=['GET'])
def resolve_vendor(vendor_input):
    """
    Resolve vendor name using VendorResolver.
    """
    if not FEATURES.get('central_vendor_resolver', False):
        return jsonify({"error": "VendorResolver not enabled"}), 503
    
    try:
        resolver = get_vendor_resolver()
        if not resolver:
            return jsonify({"error": "VendorResolver not available"}), 503
        
        # Get resolved vendors
        resolved = resolver.resolve(vendor_input, max_results=10)
        
        # Get similar vendors with scores
        similar = resolver.get_similar_vendors(vendor_input, threshold=0.6)
        
        return jsonify({
            "input": vendor_input,
            "resolved_vendors": resolved,
            "similar_vendors": [
                {"vendor": v, "similarity": f"{s:.0%}"} 
                for v, s in similar
            ],
            "canonical": resolver.get_canonical_name(vendor_input)
        })
        
    except Exception as e:
        logger.error(f"Error resolving vendor: {e}")
        return jsonify({"error": str(e)}), 500

# ============================================
# COMPARISON ENDPOINTS (WITH GROUNDED PROMPTS)
# ============================================

@app.route('/compare', methods=['POST'])
def compare():
    """
    Compare vendors with grounded comparison prompts and template parsing.
    """
    try:
        data = request.json
        vendors = data.get('vendors', [])
        
        if len(vendors) < 2:
            return jsonify({
                "error": INSUFFICIENT_DATA_MESSAGES['insufficient_vendors'].format(
                    required=2, found=len(vendors)
                )
            }), 400
        
        # Use SQL comparison with VendorResolver
        result = compare_vendors_sql(vendors)
        
        # Extract template content from narrative if present
        if 'comparison_narrative' in result:
            result['comparison_narrative'] = extract_text_from_response(result['comparison_narrative'])
        
        # Add visualization
        result['visualization'] = generate_comparison_visualization(result)
        
        # Check data sufficiency
        sufficient, message = check_data_sufficiency(result, 'comparison')
        if not sufficient:
            result['warning'] = message
        
        result['template_parsing'] = FEATURES.get('template_parsing', False)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in comparison: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/compare-semantic', methods=['POST'])
def compare_semantic():
    """
    Semantic comparison using enhanced RAG with grounded prompts and templates.
    """
    if not ENHANCED_RAG_AVAILABLE:
        return jsonify({"error": "Semantic comparison requires RAG components"}), 503
    
    try:
        data = request.json
        vendors = data.get('vendors', [])
        
        # Use semantic comparison
        result = compare_vendors_semantic(vendors)
        
        # Extract template content
        if 'answer' in result:
            result['answer'] = extract_text_from_response(result['answer'])
        if 'summary' in result:
            result['summary'] = extract_text_from_response(result['summary'])
        
        result['template_parsing'] = FEATURES.get('template_parsing', False)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in semantic comparison: {e}")
        return jsonify({"error": str(e)}), 500

# ============================================
# STATISTICS ENDPOINTS
# ============================================

@app.route('/statistics/<metric>', methods=['POST'])
def statistics(metric):
    """
    Calculate statistics with grounded statistical prompts and templates.
    """
    try:
        # Get filter criteria
        data = request.json or {}
        vendors = data.get('vendors', [])
        
        # Build query
        query = f"SELECT CAST({COST_COL} AS FLOAT) as value FROM procurement WHERE {COST_COL} IS NOT NULL"
        params = []
        
        # Apply vendor filter if specified (using VendorResolver)
        if vendors and FEATURES.get('central_vendor_resolver', False):
            resolver = get_vendor_resolver()
            resolved_vendors = []
            for vendor in vendors:
                resolved = resolver.resolve(vendor, max_results=5)
                resolved_vendors.extend(resolved)
            
            if resolved_vendors:
                placeholders = ','.join(['?' for _ in resolved_vendors])
                query += f" AND {VENDOR_COL} IN ({placeholders})"
                params = resolved_vendors
        
        df = safe_execute_query(query, params) if params else safe_execute_query(query)
        
        if df.empty:
            return jsonify({
                "error": INSUFFICIENT_DATA_MESSAGES['insufficient_samples'].format(
                    required=MIN_DATA_REQUIREMENTS.get('statistical', 10),
                    found=0
                )
            }), 404
        
        values = df['value'].dropna().values
        
        # Check data sufficiency
        if len(values) < MIN_DATA_REQUIREMENTS.get('statistical', 10):
            return jsonify({
                "warning": INSUFFICIENT_DATA_MESSAGES['insufficient_samples'].format(
                    required=MIN_DATA_REQUIREMENTS.get('statistical', 10),
                    found=len(values)
                ),
                "partial_results": calculate_statistical_metrics(values, metric)
            }), 206  # Partial Content
        
        # Calculate metrics
        result = calculate_statistical_metrics(values, metric)
        
        # Add interpretation (uses grounded prompts with templates)
        interpretation = interpret_statistics(result, metric)
        result['interpretation'] = extract_text_from_response(interpretation)
        
        result['significance'] = assess_statistical_significance(result)
        result['visualization_suggestion'] = suggest_visualization(result, metric)
        result['template_parsing'] = FEATURES.get('template_parsing', False)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error calculating statistics: {e}")
        return jsonify({"error": str(e)}), 500

# ============================================
# REPORTING ENDPOINTS
# ============================================

@app.route('/report', methods=['POST'])
def generate_report():
    """
    Generate comprehensive report with grounded data and template parsing.
    """
    if not LLM_DECOMPOSER_AVAILABLE:
        return jsonify({"error": "Report generation requires LLM components"}), 503
    
    try:
        data = request.json
        report_type = data.get('type', 'executive')
        period = data.get('period', 'all')
        focus_areas = data.get('focus_areas', ['spending', 'vendors', 'efficiency'])
        
        report = {
            'type': report_type,
            'period': period,
            'generated_at': datetime.now().isoformat(),
            'sections': {},
            'optimizations_used': {
                'grounded_prompts': FEATURES.get('grounded_prompts', False),
                'vendor_resolver': FEATURES.get('central_vendor_resolver', False),
                'template_parsing': FEATURES.get('template_parsing', False)
            }
        }
        
        # Generate executive summary
        dashboard_data = {'summary': get_dashboard_summary()}
        summary = generate_executive_summary(dashboard_data)
        report['sections']['executive_summary'] = {
            "title": "Executive Summary",
            "content": extract_text_from_response(summary)
        }
        
        # Generate sections for each focus area
        for area in focus_areas:
            section = generate_report_section(area, period)
            # Extract template content from section
            if 'content' in section:
                section['content'] = extract_text_from_response(section['content'])
            report['sections'][area] = section
        
        # Generate conclusions and recommendations
        conclusions = generate_report_conclusions(report)
        recommendations = generate_report_recommendations(report)
        
        report['sections']['conclusions'] = {
            "title": "Conclusions",
            "content": extract_text_from_response(conclusions)
        }
        report['sections']['recommendations'] = {
            "title": "Recommendations",
            "content": extract_text_from_response(recommendations)
        }
        
        # Add visualizations
        report['visualizations'] = generate_report_visualizations(report)
        
        return jsonify(report)
        
    except Exception as e:
        logger.error(f"Report generation error: {e}")
        return jsonify({"error": str(e)}), 500

# ============================================
# INSIGHTS ENDPOINTS (WITH GROUNDED DATA)
# ============================================

@app.route('/insights', methods=['POST'])
def insights():
    """
    Generate insights using grounded prompts and template parsing.
    """
    try:
        data = request.json
        focus_area = data.get('focus', 'vendors')
        
        insights_data = {
            'focus_area': focus_area,
            'insights': [],
            'generated_with': {
                'grounded_prompts': FEATURES.get('grounded_prompts', False),
                'vendor_resolver': FEATURES.get('central_vendor_resolver', False),
                'template_parsing': FEATURES.get('template_parsing', False)
            }
        }
        
        # Generate insights based on focus area
        if focus_area == 'vendors':
            raw_insights = generate_vendor_insights()
        elif focus_area == 'spending':
            raw_insights = generate_spending_insights()
        elif focus_area == 'efficiency':
            raw_insights = generate_efficiency_insights()
        else:
            # General insights
            raw_insights = generate_vendor_insights()
        
        # Extract template content from each insight
        insights_data['insights'] = [
            extract_text_from_response(insight) for insight in raw_insights
        ]
        
        return jsonify(insights_data)
        
    except Exception as e:
        logger.error(f"Error generating insights: {e}")
        return jsonify({"error": str(e)}), 500

# ============================================
# DASHBOARD ENDPOINTS
# ============================================

@app.route('/dashboard', methods=['GET'])
def dashboard():
    """
    Get dashboard summary with all metrics.
    """
    try:
        summary = get_dashboard_summary()
        
        if not summary:
            return jsonify({"error": "Unable to retrieve dashboard data"}), 500
        
        # Generate recommendations with template extraction
        raw_recommendations = generate_dashboard_recommendations()
        recommendations = [extract_text_from_response(rec) for rec in raw_recommendations]
        
        dashboard_data = {
            "summary": summary,
            "trends": get_trend_data(),
            "alerts": generate_alerts(),
            "recommendations": recommendations,
            "cache_stats": get_cache_stats() if CACHE_AVAILABLE else {},
            "features_enabled": {
                'smart_routing': FEATURES.get('smart_routing', False),
                'unified_analysis': FEATURES.get('unified_analysis', False),
                'vendor_resolver': FEATURES.get('central_vendor_resolver', False),
                'grounded_prompts': FEATURES.get('grounded_prompts', False),
                'granular_caching': FEATURES.get('granular_caching', False),
                'template_parsing': FEATURES.get('template_parsing', False)
            }
        }
        
        return jsonify(dashboard_data)
        
    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/recommend', methods=['POST'])
def recommend():
    """
    Generate strategic recommendations using a robust, SQL-grounded approach.
    """
    try:
        data = request.json
        if not data or 'context' not in data:
            return jsonify({"error": "Missing 'context' in request"}), 400
        
        context = data.get('context')
        
        # Use the robust, SQL-grounded recommendation function from app_helpers
        recommendations = get_strategic_recommendations(context)
        
        recommendations['template_parsing'] = FEATURES.get('template_parsing', False)
        
        return jsonify(recommendations)
        
    except Exception as e:
        logger.error(f"Recommendation error: {e}")
        return jsonify({"error": str(e)}), 500

# ============================================
# SYSTEM ENDPOINTS
# ============================================

@app.route('/health', methods=['GET'])
def health():
    """
    Health check with component status.
    """
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {}
    }
    
    # Check database (CRITICAL - must work)
    try:
        with db_manager.get_connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM procurement").fetchone()[0]
            health_status["components"]["database"] = {
                "status": "healthy",
                "records": count
            }
    except Exception as e:
        health_status["status"] = "unhealthy"
        health_status["components"]["database"] = {
            "status": "unhealthy",
            "error": str(e)
        }
    
    # Check LLM components
    health_status["components"]["llm_decomposer"] = {
        "status": "available" if LLM_DECOMPOSER_AVAILABLE else "not available",
        "unified_analysis": FEATURES.get('unified_analysis', False),
        "template_parsing": FEATURES.get('template_parsing', False)
    }
    
    # Check RAG
    health_status["components"]["enhanced_rag"] = {
        "status": "available" if ENHANCED_RAG_AVAILABLE else "not available",
        "grounded_prompts": FEATURES.get('grounded_prompts', False),
        "template_parsing": FEATURES.get('template_parsing', False)
    }
    
    # Check VendorResolver
    if FEATURES.get('central_vendor_resolver', False):
        try:
            resolver = get_vendor_resolver()
            health_status["components"]["vendor_resolver"] = {
                "status": "healthy" if resolver else "not available"
            }
        except:
            health_status["components"]["vendor_resolver"] = {"status": "error"}
    
    # Check cache
    if CACHE_AVAILABLE:
        try:
            cache_manager = get_cache_manager()
            cache_health = cache_manager.get_cache_health()
            health_status["components"]["cache"] = cache_health
        except:
            health_status["components"]["cache"] = {"status": "error"}
    
    # Overall status
    if any(c.get("status") in ["unhealthy", "error"] for c in health_status["components"].values()):
        health_status["status"] = "unhealthy"
        return jsonify(health_status), 503
    elif any(c.get("status") == "warning" for c in health_status["components"].values()):
        health_status["status"] = "degraded"
    
    return jsonify(health_status), 200

@app.route('/performance', methods=['GET'])
def performance():
    """
    Get system performance metrics.
    """
    try:
        perf_data = get_system_performance_stats()
        
        # Add LLM stats if available
        if LLM_DECOMPOSER_AVAILABLE:
            perf_data['llm_performance'] = get_llm_stats()
        
        # Add cache stats if available
        if CACHE_AVAILABLE:
            perf_data['cache_performance'] = get_cache_stats()
        
        return jsonify(perf_data)
        
    except Exception as e:
        logger.error(f"Performance metrics error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/cache/stats', methods=['GET'])
def cache_stats():
    """
    Get detailed cache statistics.
    """
    if not CACHE_AVAILABLE:
        return jsonify({"error": "Cache not available"}), 503
    
    try:
        cache_type = request.args.get('type')
        stats = get_cache_stats(cache_type)
        return jsonify(stats)
        
    except Exception as e:
        logger.error(f"Cache stats error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/cache/optimize', methods=['POST'])
def cache_optimize():
    """
    Optimize caches by removing expired entries.
    """
    if not CACHE_AVAILABLE:
        return jsonify({"error": "Cache not available"}), 503
    
    try:
        results = optimize_caches()
        return jsonify({
            "status": "optimized",
            "results": results
        })
        
    except Exception as e:
        logger.error(f"Cache optimization error: {e}")
        return jsonify({"error": str(e)}), 500

# ============================================
# CHAT ENDPOINT
# ============================================

@app.route('/chat', methods=['POST'])
def chat():
    """
    Conversational interface using all optimizations including template parsing.
    """
    try:
        data = request.json
        message = data.get('message')
        session_id = data.get('session_id', 'default')
        
        if not message:
            return jsonify({"error": "Missing 'message'"}), 400
        
        # Process using main pipeline
        response = process_conversational_query(message, session_id)
        
        # Extract template content
        response_text = extract_text_from_response(response)
        
        return jsonify({
            'response': response_text,
            'session_id': session_id,
            'template_parsing': FEATURES.get('template_parsing', False)
        })
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({"error": str(e)}), 500

# ============================================
# TOP VENDORS ENDPOINT
# ============================================

@app.route('/top-vendors', methods=['GET'])
def top_vendors():
    """
    Get top vendors with insights.
    """
    try:
        n = request.args.get('n', 10, type=int)
        n = min(max(n, 1), 100)
        
        query = f"""
        SELECT 
            {VENDOR_COL} as vendor,
            COUNT(*) as order_count,
            SUM(CAST({COST_COL} AS FLOAT)) as total_spending
        FROM procurement
        WHERE {COST_COL} IS NOT NULL
        GROUP BY {VENDOR_COL}
        ORDER BY total_spending DESC
        LIMIT ?
        """
        
        df = safe_execute_query(query, [n])
        
        result = {
            "count": len(df),
            "vendors": df.to_dict('records')
        }
        
        # Add insights if LLM available
        if LLM_DECOMPOSER_AVAILABLE:
            raw_insights = generate_vendor_insights_from_data(df)
            raw_recommendations = generate_vendor_recs_from_data(df)
            
            # Extract template content
            result['insights'] = extract_text_from_response(raw_insights)
            result['recommendations'] = [
                extract_text_from_response(rec) for rec in raw_recommendations
            ]
            result['template_parsing'] = FEATURES.get('template_parsing', False)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Top vendors error: {e}")
        return jsonify({"error": str(e)}), 500

# ============================================
# ROOT ENDPOINT
# ============================================

@app.route('/', methods=['GET'])
def root():
    """
    Root endpoint with API information.
    """
    return jsonify({
        "name": "Procurement RAG API",
        "version": "3.1",
        "status": "running",
        "optimizations": {
            "smart_routing": FEATURES.get('smart_routing', False),
            "unified_analysis": FEATURES.get('unified_analysis', False),
            "vendor_resolver": FEATURES.get('central_vendor_resolver', False),
            "grounded_prompts": FEATURES.get('grounded_prompts', False),
            "granular_caching": FEATURES.get('granular_caching', False),
            "tiered_search": FEATURES.get('tiered_search', False),
            "template_parsing": FEATURES.get('template_parsing', False)
        },
        "endpoints": {
            "query": "/ask",
            "advanced": "/ask-advanced",
            "vendor": "/vendor/<name>",
            "compare": "/compare",
            "statistics": "/statistics/<metric>",
            "insights": "/insights",
            "dashboard": "/dashboard",
            "health": "/health",
            "performance": "/performance"
        }
    })

# ============================================
# MAIN ENTRY POINT
# ============================================

if __name__ == '__main__':
    port = int(os.getenv('PORT', DEFAULT_PORT))
    
    print("=" * 70)
    print("  PROCUREMENT RAG API - OPTIMIZED PRODUCTION SERVER v3.1")
    print("=" * 70)
    print(f"  Port: {port}")
    print(f"  Host: {DEFAULT_HOST}")
    print("=" * 70)
    print("  OPTIMIZATIONS STATUS:")
    print(f"  ✓ Smart Routing:      {'✅ ENABLED' if FEATURES.get('smart_routing') else '❌ DISABLED'}")
    print(f"  ✓ Unified Analysis:   {'✅ ENABLED' if FEATURES.get('unified_analysis') else '❌ DISABLED'}")
    print(f"  ✓ VendorResolver:     {'✅ ENABLED' if FEATURES.get('central_vendor_resolver') else '❌ DISABLED'}")
    print(f"  ✓ Grounded Prompts:   {'✅ ENABLED' if FEATURES.get('grounded_prompts') else '❌ DISABLED'}")
    print(f"  ✓ Granular Caching:   {'✅ ENABLED' if FEATURES.get('granular_caching') else '❌ DISABLED'}")
    print(f"  ✓ Tiered Search:      {'✅ ENABLED' if FEATURES.get('tiered_search') else '❌ DISABLED'}")
    print(f"  ✓ Template Parsing:   {'✅ ENABLED' if FEATURES.get('template_parsing') else '❌ DISABLED'}")
    print("=" * 70)
    print("  COMPONENTS:")
    print(f"  • LLM Decomposer:     {'✅ AVAILABLE' if LLM_DECOMPOSER_AVAILABLE else '❌ NOT AVAILABLE'}")
    print(f"  • Enhanced RAG:       {'✅ AVAILABLE' if ENHANCED_RAG_AVAILABLE else '❌ NOT AVAILABLE'}")
    print(f"  • Cache System:       {'✅ AVAILABLE' if CACHE_AVAILABLE else '❌ NOT AVAILABLE'}")
    print(f"  • CORS Support:       {'✅ ENABLED' if CORS_AVAILABLE else '❌ DISABLED'}")
    print("=" * 70)
    print("  PERFORMANCE TARGETS:")
    print(f"  • Simple Query:       < {PERFORMANCE_TARGETS['simple_query']}s")
    print(f"  • SQL Query:          < {PERFORMANCE_TARGETS['sql_query']}s")
    print(f"  • Semantic Query:     < {PERFORMANCE_TARGETS['semantic_query']}s")
    print(f"  • Complex Query:      < {PERFORMANCE_TARGETS['complex_query']}s")
    print("=" * 70)
    print("\n  To enable optimizations, set environment variables:")
    print("  export ENABLE_SMART_ROUTING=true")
    print("  export ENABLE_UNIFIED_ANALYSIS=true")
    print("  export ENABLE_CENTRAL_RESOLVER=true")
    print("  export ENABLE_GROUNDED_PROMPTS=true")
    print("  export ENABLE_GRANULAR_CACHING=true")
    print("  export ENABLE_TIERED_SEARCH=true")
    print("  export ENABLE_TEMPLATE_PARSING=true")
    print("\n  Server starting...")
    print("=" * 70)
    
    # Run the app
    app.run(host=DEFAULT_HOST, port=port, debug=False)
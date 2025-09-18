"""
app_helpers.py - Helper functions for app.py
UPDATED: Support for template-based prompts and dual parsing modes
"""
import pandas as pd
import numpy as np
import logging
import re
import json
from typing import Dict, Any, List, Optional, Tuple

from constants import (
    VENDOR_COL, COST_COL, DESC_COL, COMMODITY_COL, DATE_COL,
    # Import new dynamic prompt functions
    get_grounded_synthesis_prompt, get_grounded_recommendation_prompt,
    get_grounded_comparison_prompt, get_grounded_statistical_prompt,
    # Import template prompts for direct use where needed
    GROUNDED_SYNTHESIS_PROMPT_TEMPLATE, GROUNDED_RECOMMENDATION_PROMPT_TEMPLATE,
    GROUNDED_COMPARISON_PROMPT_TEMPLATE, GROUNDED_STATISTICAL_PROMPT_TEMPLATE,
    # Import thresholds and messages
    MIN_DATA_REQUIREMENTS, INSUFFICIENT_DATA_MESSAGES,
    QUALITY_THRESHOLDS, FEATURES
)
from database_utils import safe_execute_query

# Configure logging
logger = logging.getLogger(__name__)

# Import VendorResolver if available
try:
    from hybrid_rag_architecture import get_vendor_resolver
    VENDOR_RESOLVER_AVAILABLE = True
except ImportError:
    VENDOR_RESOLVER_AVAILABLE = False
    logger.warning("VendorResolver not available")

# Import core AI functions
try:
    from hybrid_rag_logic import answer_question_intelligent
    from query_decomposer import generate_response, get_llm_chain
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    logger.warning("LLM components not available")

# ============================================
# ENHANCED RESPONSE HANDLING FOR TEMPLATES
# ============================================

# Import template extraction utilities
from template_utils import extract_from_template_response

def extract_text_from_response(response: Any) -> str:
    """
    Enhanced extraction for both template and JSON responses.
    Handles dict, string, template formats, and other types safely.
    """
    if response is None:
        return ""
    
    if isinstance(response, str):
        # Check if it's a template response
        if FEATURES.get('template_parsing', False):
            extracted = extract_from_template_response(response)
            if extracted and extracted != response:
                return extracted
        return response
    elif isinstance(response, dict):
        # Try common fields that might contain the text response
        for field in ['answer', 'text', 'content', 'response', 'result']:
            if field in response and response[field]:
                # Recursively extract in case the field contains template
                return extract_text_from_response(response[field])
        # Fallback to string representation
        return str(response)
    else:
        return str(response)

def format_llm_response_as_list(response: Any) -> List[str]:
    """
    Format LLM response as a list of strings.
    Enhanced to handle template responses.
    """
    text = extract_text_from_response(response)
    if not text:
        return []
    
    # Split by newlines and clean up
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    # Remove bullet points or numbering if present
    cleaned_lines = []
    for line in lines:
        # Remove common bullet point formats
        line = line.lstrip('- ').lstrip('* ').lstrip('• ')
        # Remove numbering like "1. " or "1) "
        line = re.sub(r'^\d+[\.\)]\s*', '', line)
        if line:
            cleaned_lines.append(line)
    
    return cleaned_lines

# ============================================
# VENDOR RESOLUTION HELPER (Unchanged)
# ============================================

def resolve_vendor_name(vendor_input: str) -> Optional[str]:
    """
    Resolve vendor name using VendorResolver.
    Centralized function to avoid duplication.
    """
    if VENDOR_RESOLVER_AVAILABLE and FEATURES.get('central_vendor_resolver', False):
        resolver = get_vendor_resolver()
        if resolver:
            canonical = resolver.get_canonical_name(vendor_input)
            if canonical:
                logger.info(f"Resolved '{vendor_input}' to '{canonical}'")
                return canonical
    return vendor_input

def resolve_vendor_list(vendor_input: str, max_results: int = 5) -> List[str]:
    """
    Get list of matching vendors using VendorResolver.
    """
    if VENDOR_RESOLVER_AVAILABLE and FEATURES.get('central_vendor_resolver', False):
        resolver = get_vendor_resolver()
        if resolver:
            matches = resolver.resolve(vendor_input, max_results=max_results)
            if matches:
                logger.info(f"Found {len(matches)} matches for '{vendor_input}'")
                return matches
    return [vendor_input]

# ============================================
# VENDOR DATA FUNCTIONS (Unchanged)
# ============================================

def get_vendor_comprehensive_data(vendor_name: str) -> Optional[Dict]:
    """
    Get comprehensive vendor data with VendorResolver integration.
    """
    resolved_vendors = resolve_vendor_list(vendor_name, max_results=10)
    
    if not resolved_vendors:
        logger.warning(f"No vendors found matching '{vendor_name}'")
        return None
    
    try:
        placeholders = ','.join(['?' for _ in resolved_vendors])
        
        query = f"""
        SELECT 
            MIN({VENDOR_COL}) as vendor,
            COUNT(*) as order_count,
            SUM(CAST({COST_COL} AS FLOAT)) as total_spending,
            AVG(CAST({COST_COL} AS FLOAT)) as avg_order,
            MIN(CAST({COST_COL} AS FLOAT)) as min_order,
            MAX(CAST({COST_COL} AS FLOAT)) as max_order
        FROM procurement
        WHERE {VENDOR_COL} IN ({placeholders})
        AND {COST_COL} IS NOT NULL
        GROUP BY {VENDOR_COL}
        """
        
        df = safe_execute_query(query, resolved_vendors)
        
        if not df.empty:
            result = df.iloc[0].to_dict()
            result['vendor'] = result.get('vendor', resolved_vendors[0])
            result['resolved_from'] = vendor_name
            return result
        
        return None
        
    except Exception as e:
        logger.error(f"Failed to get vendor data: {e}")
        return None

# ============================================
# INSIGHT GENERATION WITH TEMPLATE SUPPORT
# ============================================

def generate_vendor_insights() -> List[str]:
    """Generate insights about vendors using grounded data with template support."""
    try:
        query = f"""
        SELECT {VENDOR_COL}, COUNT(*) as count, SUM(CAST({COST_COL} AS FLOAT)) as total
        FROM procurement WHERE {COST_COL} IS NOT NULL
        GROUP BY {VENDOR_COL} ORDER BY total DESC LIMIT 10
        """
        df = safe_execute_query(query)
        
        if df.empty:
            return ["No vendor data available."]
        
        if LLM_AVAILABLE and FEATURES.get('grounded_prompts', False):
            data_dict = {
                'top_vendors': df.to_dict('records'),
                'total_spending': float(df['total'].sum()),
                'vendor_count': len(df)
            }
            
            context = f"""Top 10 Vendors Data:
Total Spending: ${data_dict['total_spending']:,.2f}
Number of Vendors: {data_dict['vendor_count']}

Vendor Details:
"""
            for i, vendor in enumerate(data_dict['top_vendors'], 1):
                context += f"{i}. {vendor[VENDOR_COL]}: ${vendor['total']:,.2f} ({vendor['count']} orders)\n"
            
            # Use dynamic prompt function
            prompt = get_grounded_synthesis_prompt().format(
                context=context,
                question="Generate three key business insights from this vendor data."
            )
            
            response = generate_response("vendor insights", {'context': context})
            # Use enhanced extraction
            return format_llm_response_as_list(response)
        else:
            # Fallback to rule-based insights
            insights = []
            top_vendor = df.iloc[0]
            insights.append(f"Top vendor {top_vendor[VENDOR_COL]} accounts for ${top_vendor['total']:,.2f}")
            
            total_spending = df['total'].sum()
            top_3_spending = df.head(3)['total'].sum()
            concentration = (top_3_spending / total_spending * 100) if total_spending > 0 else 0
            
            if concentration > 60:
                insights.append(f"High vendor concentration: Top 3 vendors = {concentration:.1f}% of spending")
            
            return insights
            
    except Exception as e:
        logger.error(f"Error generating vendor insights: {e}")
        return [f"Error generating vendor insights: {e}"]

def generate_spending_insights() -> List[str]:
    """Generate insights about spending patterns using grounded data with template support."""
    try:
        query = f"""
        SELECT COUNT(*) as count, SUM(CAST({COST_COL} AS FLOAT)) as total,
        AVG(CAST({COST_COL} AS FLOAT)) as avg, MIN(CAST({COST_COL} AS FLOAT)) as min,
        MAX(CAST({COST_COL} AS FLOAT)) as max
        FROM procurement WHERE {COST_COL} IS NOT NULL
        """
        df = safe_execute_query(query)
        
        if df.empty:
            return ["No spending data available."]
        
        row = df.iloc[0]
        
        if LLM_AVAILABLE and FEATURES.get('grounded_prompts', False):
            statistics = {
                'total_spending': float(row['total']),
                'order_count': int(row['count']),
                'average_order': float(row['avg']),
                'min_order': float(row['min']),
                'max_order': float(row['max'])
            }
            
            # Use dynamic prompt function for statistical analysis
            prompt = get_grounded_statistical_prompt().format(
                statistics=json.dumps(statistics),
                question="What are the key insights from these spending statistics?"
            )
            
            response = generate_response("spending insights", {'statistics': statistics})
            return format_llm_response_as_list(response)
        else:
            insights = []
            insights.append(f"Total spending: ${row['total']:,.2f} across {row['count']} orders")
            
            if row['max'] > row['avg'] * 10:
                insights.append(f"High-value outlier detected: ${row['max']:,.2f}")
            
            return insights
            
    except Exception as e:
        logger.error(f"Error generating spending insights: {e}")
        return [f"Error generating spending insights: {e}"]

def generate_efficiency_insights() -> List[str]:
    """Generate insights about procurement efficiency."""
    try:
        query = f"SELECT COUNT(DISTINCT {VENDOR_COL}) as vendor_count, COUNT(*) as order_count FROM procurement"
        df = safe_execute_query(query)
        
        if df.empty:
            return ["No efficiency data available."]
        
        row = df.iloc[0]
        vendor_count = int(row['vendor_count'])
        order_count = int(row['order_count'])
        
        orders_per_vendor = order_count / vendor_count if vendor_count > 0 else 0
        
        insights = []
        if orders_per_vendor < 5:
            insights.append(f"Low vendor utilization: {orders_per_vendor:.1f} orders per vendor. Consider consolidation.")
        else:
            insights.append(f"Vendor utilization: {orders_per_vendor:.1f} orders per vendor on average.")
        
        insights.append(f"Total of {vendor_count} unique vendors managing {order_count} orders.")
        
        return insights
        
    except Exception as e:
        logger.error(f"Error generating efficiency insights: {e}")
        return [f"Error generating efficiency insights: {e}"]

# ============================================
# VENDOR ANALYSIS FUNCTIONS WITH TEMPLATE SUPPORT
# ============================================

def generate_vendor_analysis(vendor_data: Dict) -> str:
    """Generate analysis text for a vendor using grounded prompts with template support."""
    if not vendor_data:
        return "No vendor data available for analysis"
    
    if LLM_AVAILABLE and FEATURES.get('grounded_prompts', False):
        context = f"""Vendor: {vendor_data.get('vendor')}
Total Orders: {vendor_data.get('order_count', 0):,}
Total Spending: ${vendor_data.get('total_spending', 0):,.2f}
Average Order: ${vendor_data.get('avg_order', 0):,.2f}
Order Range: ${vendor_data.get('min_order', 0):,.2f} - ${vendor_data.get('max_order', 0):,.2f}"""
        
        # Use dynamic prompt function
        prompt = get_grounded_synthesis_prompt().format(
            context=context,
            question="Provide a brief analysis of this vendor's performance."
        )
        
        response = generate_response("vendor analysis", {'vendor_data': vendor_data})
        return extract_text_from_response(response)
    else:
        return f"""Vendor Analysis for {vendor_data.get('vendor')}:
- Total Orders: {vendor_data.get('order_count', 0):,}
- Total Spending: ${vendor_data.get('total_spending', 0):,.2f}
- Average Order: ${vendor_data.get('avg_order', 0):,.2f}"""

def assess_vendor_risk(vendor_data: Dict) -> str:
    """Assess vendor risk level using data-driven analysis."""
    if not vendor_data:
        return "Unable to assess risk - no data"
    
    order_count = vendor_data.get('order_count', 0)
    avg_order = vendor_data.get('avg_order', 0)
    
    risk_factors = []
    risk_level = "Low"
    
    if order_count < MIN_DATA_REQUIREMENTS.get('recommendation', 5):
        risk_factors.append("Low order volume")
        risk_level = "Medium"
    
    if avg_order > 100000:
        risk_factors.append("High value orders")
        if risk_level == "Medium":
            risk_level = "High"
        else:
            risk_level = "Medium"
    
    if order_count == 1:
        risk_factors.append("Single order vendor")
        risk_level = "High"
    
    if risk_factors:
        return f"Risk Level: {risk_level} - Factors: {', '.join(risk_factors)}"
    else:
        return f"Risk Level: {risk_level} - Stable vendor relationship"

def identify_vendor_opportunities(vendor_data: Dict) -> str:
    """Identify opportunities with a vendor based on data."""
    if not vendor_data:
        return "No data for opportunity analysis"
    
    opportunities = []
    
    order_count = vendor_data.get('order_count', 0)
    total_spending = vendor_data.get('total_spending', 0)
    avg_order = vendor_data.get('avg_order', 0)
    
    if order_count > 20:
        opportunities.append("Volume discount negotiation (high order count)")
    
    if total_spending > 500000:
        opportunities.append("Strategic partnership opportunity (high spending)")
    
    if avg_order > 50000:
        opportunities.append("Bulk order optimization (high average order value)")
    
    if order_count < 5:
        opportunities.append("Vendor consolidation candidate (low activity)")
    
    if opportunities:
        return "Opportunities: " + "; ".join(opportunities)
    else:
        return "No immediate opportunities identified"

# ============================================
# COMPREHENSIVE VENDOR ANALYSIS (Unchanged)
# ============================================

def analyze_vendor_comprehensive(vendor_name: str) -> Dict:
    """
    Comprehensive vendor analysis with VendorResolver.
    """
    vendor_data = get_vendor_comprehensive_data(vendor_name)
    
    if not vendor_data:
        if VENDOR_RESOLVER_AVAILABLE and FEATURES.get('central_vendor_resolver', False):
            resolver = get_vendor_resolver()
            if resolver:
                similar = resolver.get_similar_vendors(vendor_name, threshold=0.6)
                if similar:
                    suggestions = [f"{vendor} (similarity: {score:.0%})" for vendor, score in similar[:3]]
                    return {
                        "error": f"Vendor '{vendor_name}' not found.",
                        "suggestions": suggestions,
                        "message": f"Did you mean one of these: {', '.join([v for v, _ in similar[:3]])}?"
                    }
        
        return {"error": f"Vendor '{vendor_name}' not found."}
    
    analysis = {
        "vendor_data": vendor_data,
        "analysis": generate_vendor_analysis(vendor_data),
        "risk_assessment": assess_vendor_risk(vendor_data),
        "opportunities": identify_vendor_opportunities(vendor_data)
    }
    
    commodity_data = get_vendor_commodities(vendor_data.get('vendor'))
    if commodity_data:
        analysis['top_categories'] = commodity_data[:5]
    
    return analysis

def get_vendor_commodities(vendor_name: str, limit: int = 10) -> List[Dict]:
    """Get commodity breakdown for a vendor."""
    if not vendor_name:
        return []
    
    resolved_vendors = resolve_vendor_list(vendor_name, max_results=10)
    
    if not resolved_vendors:
        return []
    
    try:
        placeholders = ','.join(['?' for _ in resolved_vendors])
        
        query = f"""
        SELECT 
            {COMMODITY_COL} as commodity,
            COUNT(*) as count,
            SUM(CAST({COST_COL} AS FLOAT)) as total
        FROM procurement
        WHERE {VENDOR_COL} IN ({placeholders})
        AND {COMMODITY_COL} IS NOT NULL
        AND {COST_COL} IS NOT NULL
        GROUP BY {COMMODITY_COL}
        ORDER BY total DESC
        LIMIT ?
        """
        
        params = resolved_vendors + [limit]
        df = safe_execute_query(query, params)
        
        return df.to_dict('records') if not df.empty else []
        
    except Exception as e:
        logger.error(f"Failed to get commodity data: {e}")
        return []

# ============================================
# COMPARISON FUNCTIONS WITH TEMPLATE SUPPORT
# ============================================

def compare_vendors_sql(vendors: List[str], metrics: List[str] = None) -> Dict:
    """Compare vendors using SQL with VendorResolver and template support."""
    if not metrics:
        metrics = ['total_spending', 'order_count', 'avg_order']
    
    results = {"vendors": [], "comparison_metrics": metrics}
    
    for vendor in vendors[:10]:
        vendor_data = get_vendor_comprehensive_data(vendor)
        if vendor_data:
            filtered_data = {k: v for k, v in vendor_data.items() if k in metrics or k == 'vendor'}
            results["vendors"].append(filtered_data)
    
    if results["vendors"] and LLM_AVAILABLE and FEATURES.get('grounded_prompts', False):
        vendor_data_str = "\n".join([
            f"{v['vendor']}: Total=${v.get('total_spending', 0):,.2f}, Orders={v.get('order_count', 0)}, Avg=${v.get('avg_order', 0):,.2f}"
            for v in results["vendors"]
        ])
        
        # Use dynamic prompt function
        prompt = get_grounded_comparison_prompt().format(
            vendor_data=vendor_data_str,
            question=f"Compare these vendors: {', '.join(vendors)}"
        )
        
        response = generate_response("vendor comparison", results)
        results["comparison_narrative"] = extract_text_from_response(response)
    
    return results

def generate_comparison_visualization(comparison_result: Dict) -> Dict:
    """Generate visualization data for comparison."""
    viz_data = {
        "type": "bar",
        "data": {
            "labels": [],
            "datasets": [
                {"label": "Total Spending", "data": []},
                {"label": "Order Count", "data": []}
            ]
        }
    }
    
    for vendor in comparison_result.get("vendors", []):
        viz_data["data"]["labels"].append(vendor.get("vendor", "Unknown"))
        viz_data["data"]["datasets"][0]["data"].append(vendor.get("total_spending", 0))
        viz_data["data"]["datasets"][1]["data"].append(vendor.get("order_count", 0))
    
    return viz_data

# ============================================
# STATISTICAL FUNCTIONS WITH TEMPLATE SUPPORT
# ============================================

def calculate_statistical_metrics(values: np.ndarray, metric: str = "all") -> Dict:
    """Calculate statistical metrics."""
    if len(values) == 0:
        return {"error": "No values to analyze"}
    
    result = {
        "metric": metric,
        "records_analyzed": len(values)
    }
    
    if metric == "all":
        result.update({
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": float(np.std(values)),
            "variance": float(np.var(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "q25": float(np.percentile(values, 25)),
            "q75": float(np.percentile(values, 75))
        })
    else:
        stats_map = {
            "median": np.median,
            "mean": np.mean,
            "stddev": np.std,
            "variance": np.var,
            "min": np.min,
            "max": np.max
        }
        if metric in stats_map:
            result["value"] = float(stats_map[metric](values))
        else:
            result["error"] = f"Unknown metric: {metric}"
    
    return result

def interpret_statistics(result: Dict, metric: str) -> str:
    """Interpret statistical results using grounded approach with template support."""
    if "error" in result:
        return result["error"]

    def get_simple_interpretation():
        """Provides a basic, non-LLM interpretation."""
        if "value" in result:
            return f"The {metric} is ${result['value']:,.2f} based on {result['records_analyzed']} records."
        else:
            # Handle 'all' case
            mean = result.get('mean', 0)
            median = result.get('median', 0)
            return (f"Statistical analysis complete for {result['records_analyzed']} records. "
                    f"Mean: ${mean:,.2f}, Median: ${median:,.2f}.")

    if LLM_AVAILABLE and FEATURES.get('grounded_prompts', False):
        prompt_template_str = get_grounded_statistical_prompt()
        llm_question = f"Interpret these {metric} statistics in business terms. Provide a summary, key findings, and business impact."

        try:
            llm_chain = get_llm_chain(prompt_template_str)

            response = llm_chain.run({
                "statistics": json.dumps(result, indent=2, default=str),
                "question": llm_question
            })

            extracted_response = extract_text_from_response(response)

            # Per user feedback, avoid silent fallbacks. If LLM gives a bad response, log it and use the simple one.
            if not extracted_response or "insufficient data" in extracted_response.lower():
                 logger.warning(f"LLM returned a low-quality interpretation. Raw response: {response}")
                 return get_simple_interpretation()

            return extracted_response

        except Exception as e:
            logger.error(f"LLM-based statistical interpretation failed: {e}")
            return get_simple_interpretation()
    else:
        return get_simple_interpretation()

# ============================================
# DASHBOARD & REPORT FUNCTIONS WITH TEMPLATE SUPPORT
# ============================================

def get_dashboard_summary() -> Dict:
    """Get dashboard summary data."""
    try:
        query = f"""
        SELECT COUNT(*) as total_orders, COUNT(DISTINCT {VENDOR_COL}) as total_vendors,
        SUM(CAST({COST_COL} AS FLOAT)) as total_spending, AVG(CAST({COST_COL} AS FLOAT)) as avg_order
        FROM procurement WHERE {COST_COL} IS NOT NULL
        """
        df = safe_execute_query(query)
        
        if not df.empty:
            result = {}
            for key, value in df.iloc[0].to_dict().items():
                if pd.isna(value):
                    result[key] = None
                elif isinstance(value, (np.integer, np.int64, np.int32)):
                    result[key] = int(value)
                elif isinstance(value, (np.floating, np.float64, np.float32)):
                    result[key] = float(value)
                else:
                    result[key] = value
            return result
        return {}
        
    except Exception as e:
        logger.error(f"Failed to get dashboard summary: {e}")
        return {}

def generate_dashboard_recommendations() -> List[str]:
    """Generate dashboard recommendations using grounded data with template support."""
    summary = get_dashboard_summary()
    
    if not summary:
        return ["Unable to generate recommendations - no data available"]
    
    if LLM_AVAILABLE and FEATURES.get('grounded_prompts', False):
        context = f"""Dashboard Summary:
- Total Orders: {summary.get('total_orders', 0):,}
- Total Vendors: {summary.get('total_vendors', 0):,}
- Total Spending: ${summary.get('total_spending', 0):,.2f}
- Average Order: ${summary.get('avg_order', 0):,.2f}"""
        
        # Use dynamic prompt function
        prompt = get_grounded_recommendation_prompt().format(
            context=context,
            focus="procurement optimization"
        )
        
        response = generate_response("dashboard recommendations", {'summary': summary})
        return format_llm_response_as_list(response)
    else:
        recs = []
        
        if summary.get('total_vendors', 0) > 100:
            recs.append("Consider vendor consolidation - over 100 vendors active")
        
        if summary.get('avg_order', 0) < 1000:
            recs.append("Many small orders detected - consider bulk ordering")
        
        return recs if recs else ["Review spending patterns for optimization opportunities"]

# ============================================
# REPORT GENERATION FUNCTIONS WITH TEMPLATE SUPPORT
# ============================================

def generate_report_section(area: str, period: str = "all") -> Dict:
    """Generate a report section using grounded data with template support."""
    title = f"{area.title()} Analysis"
    data = {}
    
    if area == "spending":
        data = get_dashboard_summary()
    elif area == "vendors":
        data = {"insights": generate_vendor_insights()}
    elif area == "efficiency":
        data = {"insights": generate_efficiency_insights()}
    
    content = f"Analysis for {area} (Period: {period})\n"
    
    if data:
        if LLM_AVAILABLE and FEATURES.get('grounded_prompts', False):
            # Use dynamic prompt function
            prompt = get_grounded_synthesis_prompt().format(
                context=json.dumps(data) if not isinstance(data, str) else data,
                question=f"Create a brief report section about {area}"
            )
            response = generate_response(f"{area} report", data)
            content = extract_text_from_response(response)
        else:
            content += str(data)
    
    return {
        "title": title,
        "content": content,
        "data": data
    }

def generate_executive_summary(dashboard_data: Dict) -> str:
    """Generate executive summary using grounded data with template support."""
    summary_data = dashboard_data.get('summary', {})
    
    if not summary_data:
        return "Executive summary unavailable - no data"
    
    if LLM_AVAILABLE and FEATURES.get('grounded_prompts', False):
        context = f"""Key Metrics:
- Total Orders: {summary_data.get('total_orders', 0):,}
- Total Vendors: {summary_data.get('total_vendors', 0):,}
- Total Spending: ${summary_data.get('total_spending', 0):,.2f}
- Average Order Value: ${summary_data.get('avg_order', 0):,.2f}"""
        
        # Use dynamic prompt function
        prompt = get_grounded_synthesis_prompt().format(
            context=context,
            question="Write a brief executive summary of the procurement status"
        )
        
        response = generate_response("executive summary", summary_data)
        return extract_text_from_response(response)
    else:
        return f"""Executive Summary:
- Total procurement spending: ${summary_data.get('total_spending', 0):,.2f}
- Active vendors: {summary_data.get('total_vendors', 0):,}
- Total orders processed: {summary_data.get('total_orders', 0):,}
- Average order value: ${summary_data.get('avg_order', 0):,.2f}"""

# ============================================
# HELPER UTILITIES (Unchanged)
# ============================================

def process_conversational_query(message: str, session_id: str = "default") -> str:
    """Process conversational query using the main pipeline."""
    if LLM_AVAILABLE:
        result = answer_question_intelligent(message)
        return result.get('answer', "I was unable to find an answer. Please try rephrasing.")
    else:
        return "Conversational processing requires LLM components."

def check_data_sufficiency(data: Any, requirement_type: str) -> Tuple[bool, str]:
    """
    Check if data meets minimum requirements.
    """
    if requirement_type not in MIN_DATA_REQUIREMENTS:
        return True, ""
    
    required = MIN_DATA_REQUIREMENTS[requirement_type]
    
    if isinstance(data, pd.DataFrame):
        actual = len(data)
    elif isinstance(data, list):
        actual = len(data)
    elif isinstance(data, dict) and 'vendors' in data:
        actual = len(data['vendors'])
    else:
        actual = 0
    
    if actual < required:
        message = INSUFFICIENT_DATA_MESSAGES.get(
            f'insufficient_{requirement_type}',
            INSUFFICIENT_DATA_MESSAGES['no_data']
        )
        if '{required}' in message and '{found}' in message:
            message = message.format(required=required, found=actual)
        return False, message
    
    return True, ""

# ============================================
# BACKWARD COMPATIBILITY FUNCTIONS (Unchanged)
# ============================================

def enhance_insights_with_llm(insights_data: Dict, context: str = "") -> Dict:
    """Backward compatibility - insights now enhanced by default"""
    return insights_data

def generate_vendor_insights_from_data(df: pd.DataFrame) -> str:
    """Generate insights from vendor dataframe."""
    if df.empty:
        return "No vendor data available"
    
    insights = []
    insights.append(f"Total vendors analyzed: {len(df)}")
    
    if 'total_spending' in df.columns:
        total = df['total_spending'].sum()
        insights.append(f"Combined spending: ${total:,.2f}")
    
    return " | ".join(insights)

def generate_vendor_recommendations(df: pd.DataFrame) -> List[str]:
    """Generate recommendations from vendor dataframe."""
    if df.empty:
        return ["Insufficient data for recommendations"]
    
    recs = []
    
    if 'order_count' in df.columns:
        low_activity = df[df['order_count'] < 5]
        if len(low_activity) > 3:
            recs.append(f"Consider consolidating {len(low_activity)} low-activity vendors")
    
    if 'total_spending' in df.columns:
        top_vendor_spending = df.iloc[0]['total_spending'] if not df.empty else 0
        total_spending = df['total_spending'].sum()
        if total_spending > 0 and top_vendor_spending / total_spending > 0.3:
            recs.append("High concentration with top vendor - consider diversification")
    
    return recs if recs else ["Current vendor distribution appears balanced"]

# Additional backward compatibility functions
def generate_report_conclusions(report: Dict) -> str:
    """Generate report conclusions."""
    sections = report.get('sections', {})
    if not sections:
        return "No data available for conclusions"
    
    conclusions = []
    for section_name, section_data in sections.items():
        if isinstance(section_data, dict) and 'content' in section_data:
            conclusions.append(f"{section_name}: Analyzed")
    
    return "Analysis complete for: " + ", ".join(conclusions)

def generate_report_recommendations(report: Dict) -> str:
    """
    Generate strategic recommendations based on the content of the report.
    This function calls the more powerful get_strategic_recommendations for each
    section in the report to generate context-specific advice.
    """
    all_recs = []

    # Mapping from report section area to a strategic context
    context_mapping = {
        'spending': 'cost optimization',
        'vendors': 'vendor management and consolidation',
        'efficiency': 'procurement process improvement',
        'executive_summary': 'overall procurement strategy',
        'conclusions': 'strategic planning'
    }

    report_sections = report.get('sections', {})
    if not report_sections:
        return "No report sections available to generate recommendations."

    for area in report_sections.keys():
        context = context_mapping.get(area.lower())
        if context:
            logger.info(f"Generating report recommendations for area: {area} with context: {context}")
            # Get strategic recommendations for this context
            strategic_rec_dict = get_strategic_recommendations(context)

            # Extract the text answer
            rec_text = extract_text_from_response(strategic_rec_dict)

            if rec_text and "error" not in rec_text.lower() and "insufficient data" not in rec_text.lower():
                # Add a title for each section's recommendations
                all_recs.append(f"--- Recommendations for {area.replace('_', ' ').title()} ---")
                all_recs.append(rec_text)
                all_recs.append("") # Add a blank line for spacing

    if not all_recs:
        return "Could not generate specific recommendations based on the report data."

    return "\n".join(all_recs)

def generate_report_visualizations(report: Dict) -> List[Dict]:
    """Generate visualization configurations."""
    return [
        {"type": "pie", "title": "Spending by Vendor"},
        {"type": "bar", "title": "Top 10 Vendors"},
        {"type": "line", "title": "Spending Trend"}
    ]

def get_trend_data() -> Dict:
    """
    Get trend data for spending over time.
    This function queries the database to get monthly spending totals.
    """
    query = f"""
    SELECT
        strftime('%Y-%m', {DATE_COL}) as month,
        SUM(CAST({COST_COL} AS FLOAT)) as total_spending
    FROM procurement
    WHERE {DATE_COL} IS NOT NULL AND {COST_COL} IS NOT NULL
    GROUP BY month
    ORDER BY month ASC
    """

    try:
        df = safe_execute_query(query)

        if df.empty:
            return {
                "available": False,
                "message": "Insufficient data to generate trend analysis."
            }

        # Format for chart
        labels = df['month'].tolist()
        data = df['total_spending'].tolist()

        return {
            "available": True,
            "labels": labels,
            "datasets": [
                {
                    "label": "Total Spending per Month",
                    "data": data
                }
            ]
        }

    except Exception as e:
        logger.error(f"Failed to get trend data: {e}")
        return {
            "available": False,
            "message": f"An error occurred while generating trend data: {e}"
        }

def generate_alerts() -> List[Dict]:
    """Generate system alerts based on thresholds."""
    alerts = []
    
    try:
        query = f"""
        SELECT {VENDOR_COL}, {COST_COL} 
        FROM procurement
        WHERE CAST({COST_COL} AS FLOAT) > (
            SELECT AVG(CAST({COST_COL} AS FLOAT)) * 10 
            FROM procurement WHERE {COST_COL} IS NOT NULL
        )
        LIMIT 3
        """
        df = safe_execute_query(query)
        
        for _, row in df.iterrows():
            alerts.append({
                "type": "warning",
                "message": f"High-value order: {row[VENDOR_COL]} - ${float(row[COST_COL]):,.2f}"
            })
    except:
        pass
    
    return alerts

# Placeholder functions for missing implementations
def suggest_visualization(result: Dict, metric: str) -> str:
    """Suggest visualization type."""
    if metric in ["mean", "median"]:
        return "Suggested: Box plot to show distribution"
    elif metric == "variance":
        return "Suggested: Histogram to show spread"
    else:
        return "Suggested: Bar chart for comparison"

def assess_statistical_significance(result: Dict) -> str:
    """Assess statistical significance."""
    if result.get("std", 0) > result.get("mean", 1) * 0.5:
        return "High variance detected - results show significant spread"
    return "Results show normal distribution"

def analyze_vendor_sql(vendor_name: str) -> Dict:
    """SQL-only vendor analysis."""
    vendor_data = get_vendor_comprehensive_data(vendor_name)
    return vendor_data or {"error": "Vendor not found"}

def analyze_spending_patterns(df: pd.DataFrame) -> List[str]:
    """Analyze spending patterns."""
    patterns = []
    if df.empty:
        return ["No data for pattern analysis"]
    
    if 'total_spending' in df.columns:
        high_spenders = df[df['total_spending'] > df['total_spending'].mean() * 2]
        if not high_spenders.empty:
            patterns.append(f"{len(high_spenders)} vendors with above-average spending")
    
    return patterns

def analyze_optimization_opportunities() -> Dict:
    """Analyze optimization opportunities."""
    try:
        query = f"""
        SELECT COUNT(DISTINCT {VENDOR_COL}) as vendor_count,
               COUNT(*) as order_count
        FROM procurement
        """
        df = safe_execute_query(query)
        
        if not df.empty:
            row = df.iloc[0]
            opportunities = []
            
            if int(row['vendor_count']) > 50:
                opportunities.append("Vendor consolidation opportunity")
            
            return {"opportunities": opportunities}
    except:
        pass
    
    return {"opportunities": ["Analysis in progress"]}

def perform_general_analysis(question: str) -> Dict:
    """General analysis using main pipeline."""
    if LLM_AVAILABLE:
        return answer_question_intelligent(question)
    return {"error": "Analysis requires LLM components"}

def generate_sql_recommendations(df: pd.DataFrame) -> List[str]:
    """Generate SQL-based recommendations."""
    return generate_vendor_recommendations(df)

def generate_action_items(analysis_result: Dict) -> List[str]:
    """Generate action items."""
    items = []
    
    if analysis_result.get('vendor_data'):
        vendor_data = analysis_result['vendor_data']
        if vendor_data.get('order_count', 0) < 5:
            items.append(f"Review relationship with {vendor_data.get('vendor')}")
    
    return items if items else ["Continue monitoring"]

def generate_priority_matrix(action_items: List[str]) -> Dict:
    """Generate priority matrix."""
    matrix = []
    for i, item in enumerate(action_items):
        matrix.append({
            "action": item,
            "impact": ["High", "Medium", "Low"][i % 3],
            "effort": ["Low", "Medium", "High"][i % 3]
        })
    
    return {"priority_matrix": matrix}

def identify_spending_patterns(df: pd.DataFrame) -> List[str]:
    """Identify spending patterns."""
    return analyze_spending_patterns(df)

def combine_analysis_results(results: Dict) -> str:
    """Combine multiple analysis results."""
    combined = []
    for key, value in results.items():
        if isinstance(value, dict) and 'answer' in value:
            combined.append(f"{key}: {value['answer'][:100]}...")
    
    return "\n".join(combined) if combined else "Analysis complete"

def get_strategic_recommendations(context: str = "cost optimization") -> Dict[str, Any]:
    """
    Get strategic recommendations based on data from the SQL database.
    This is the primary implementation for recommendations, bypassing semantic search.
    """
    logger.info(f"Generating recommendations for context: {context}")

    try:
        top_spenders_query = f"""
        SELECT "{VENDOR_COL}" as vendor, SUM(CAST("{COST_COL}" AS FLOAT)) as total_spending, COUNT(*) as order_count
        FROM procurement
        WHERE "{COST_COL}" IS NOT NULL
        GROUP BY "{VENDOR_COL}"
        ORDER BY total_spending DESC
        LIMIT 10
        """
        top_spenders_df = safe_execute_query(top_spenders_query)

        consolidation_candidates_query = f"""
        SELECT "{VENDOR_COL}" as vendor, COUNT(*) as order_count, AVG(CAST("{COST_COL}" AS FLOAT)) as avg_order
        FROM procurement
        WHERE "{COST_COL}" IS NOT NULL
        GROUP BY "{VENDOR_COL}"
        HAVING order_count > 10 AND avg_order < 500
        ORDER BY order_count DESC
        LIMIT 10
        """
        consolidation_df = safe_execute_query(consolidation_candidates_query)

    except Exception as e:
        logger.error(f"Failed to query data for recommendations: {e}")
        return {"error": "Database query for recommendations failed.", "answer": "Could not retrieve data to generate recommendations."}

    data_context = {
        "focus_area": context,
        "top_10_vendors_by_spending": top_spenders_df.to_dict('records'),
        "potential_consolidation_candidates": consolidation_df.to_dict('records')
    }

    if top_spenders_df.empty and consolidation_df.empty:
        return {
            "answer": "I was unable to find sufficient data to generate recommendations for your specified context.",
            "records_analyzed": 0,
            "source": "SQL"
        }

    if not LLM_AVAILABLE:
        logger.warning("LLM not available, returning basic recommendations.")
        # Basic non-LLM recommendations
        recs = []
        if not consolidation_df.empty:
            recs.append(f"Consider consolidating up to {len(consolidation_df)} vendors with many small orders.")
        if not top_spenders_df.empty:
            recs.append(f"Review spending with top vendors like {top_spenders_df.iloc[0]['vendor']} for potential savings.")
        return {"answer": "\n".join(recs) if recs else "No specific recommendations generated."}

    try:
        prompt = get_grounded_recommendation_prompt().format(
            context=json.dumps(data_context, indent=2),
            focus=context,
            question=f"Generate recommendations for {context} based on the provided data."
        )

        llm_chain = get_llm_chain(prompt)
        llm_response = llm_chain.run({})
        response_text = extract_text_from_response(llm_response)

        if not response_text or "insufficient" in response_text.lower():
            response_text = "Based on the available data, no specific recommendations could be generated by the LLM."

        return {
            "answer": response_text,
            "summary": response_text,
            "records_analyzed": len(top_spenders_df) + len(consolidation_df),
            "confidence": 85,
            "source": "SQL-Grounded RAG",
            "recommendation_type": context,
            "grounded_recommendation": True,
            "template_parsing": FEATURES.get('template_parsing', False)
        }
    except Exception as e:
        logger.error(f"LLM call for recommendations failed: {e}")
        return {"error": "Failed to generate recommendations from the language model."}
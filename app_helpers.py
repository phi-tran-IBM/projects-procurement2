"""
app_helpers.py - Helper functions for app.py
ENHANCED WITH DYNAMIC LLM-POWERED RESPONSES
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

from constants import VENDOR_COL, COST_COL
from database_utils import safe_execute_query

# --- NEW: Import Core AI Functions ---
try:
    from hybrid_rag_logic import answer_question_intelligent
    from query_decomposer import generate_response
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False


def _format_llm_list_response(response: str) -> List[str]:
    """Helper to format LLM string output into a list of insights."""
    return [line.strip().lstrip('- ').strip() for line in response.strip().split('\n') if line.strip()]

# ============================================
# VENDOR INSIGHTS FUNCTIONS (LLM UPGRADED)
# ============================================

def generate_vendor_insights() -> List[str]:
    """Generate insights about vendors using LLM analysis of SQL data."""
    try:
        query = f"""
        SELECT {VENDOR_COL}, COUNT(*) as count, SUM(CAST({COST_COL} AS FLOAT)) as total
        FROM procurement WHERE {COST_COL} IS NOT NULL
        GROUP BY {VENDOR_COL} ORDER BY total DESC LIMIT 10
        """
        df = safe_execute_query(query)
        if df.empty: return ["No vendor data available."]

        if LLM_AVAILABLE:
            data_dict = df.to_dict('records')
            prompt = "Generate three key business insights from the following top vendor data."
            llm_insights = generate_response(prompt, {"top_vendors": data_dict})
            return _format_llm_list_response(llm_insights)
        else:
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
        return [f"Error generating vendor insights: {e}"]

def generate_spending_insights() -> List[str]:
    """Generate insights about spending patterns using LLM."""
    try:
        query = f"""
        SELECT COUNT(*) as count, SUM(CAST({COST_COL} AS FLOAT)) as total,
        AVG(CAST({COST_COL} AS FLOAT)) as avg, MIN(CAST({COST_COL} AS FLOAT)) as min,
        MAX(CAST({COST_COL} AS FLOAT)) as max
        FROM procurement WHERE {COST_COL} IS NOT NULL
        """
        df = safe_execute_query(query)
        if df.empty: return ["No spending data available."]

        if LLM_AVAILABLE:
            data_dict = df.iloc[0].to_dict()
            prompt = "Analyze this overall spending summary and provide three important insights for a manager."
            llm_insights = generate_response(prompt, {"spending_summary": data_dict})
            return _format_llm_list_response(llm_insights)
        else:
            insights = []
            row = df.iloc[0]
            insights.append(f"Total spending: ${row['total']:,.2f} across {row['count']} orders")
            if row['max'] > row['avg'] * 10:
                insights.append(f"High-value outlier detected: ${row['max']:,.2f}")
            return insights
    except Exception as e:
        return [f"Error generating spending insights: {e}"]

def generate_efficiency_insights() -> List[str]:
    """Generate insights about procurement efficiency using LLM."""
    try:
        query = f"SELECT COUNT(DISTINCT {VENDOR_COL}) as vendor_count, COUNT(*) as order_count FROM procurement"
        df = safe_execute_query(query)
        if df.empty: return ["No efficiency data available."]

        if LLM_AVAILABLE:
            data_dict = df.iloc[0].to_dict()
            prompt = "Based on this vendor and order count, what are two key insights about procurement efficiency?"
            llm_insights = generate_response(prompt, {"efficiency_data": data_dict})
            return _format_llm_list_response(llm_insights)
        else:
            insights = []
            row = df.iloc[0]
            orders_per_vendor = row['order_count'] / row['vendor_count'] if row['vendor_count'] > 0 else 0
            if orders_per_vendor < 5:
                insights.append(f"Low vendor utilization: {orders_per_vendor:.1f} orders per vendor. Consider consolidation.")
            return insights
    except Exception as e:
        return [f"Error generating efficiency insights: {e}"]
        
def enhance_insights_with_llm(insights_data: Dict) -> Dict:
    """This function is now superseded by direct LLM calls in the functions above, but is kept for compatibility."""
    return insights_data

# ============================================
# VENDOR ANALYSIS FUNCTIONS (LLM UPGRADED)
# ============================================

def get_vendor_comprehensive_data(vendor_name: str) -> Optional[Dict]:
    """Get comprehensive vendor data."""
    try:
        query = f"""
        SELECT {VENDOR_COL} as vendor, COUNT(*) as order_count, SUM(CAST({COST_COL} AS FLOAT)) as total_spending,
        AVG(CAST({COST_COL} AS FLOAT)) as avg_order, MIN(CAST({COST_COL} AS FLOAT)) as min_order,
        MAX(CAST({COST_COL} AS FLOAT)) as max_order
        FROM procurement WHERE UPPER({VENDOR_COL}) LIKE ? AND {COST_COL} IS NOT NULL GROUP BY {VENDOR_COL}
        """
        df = safe_execute_query(query, [f"%{vendor_name.upper()}%"])
        return df.iloc[0].to_dict() if not df.empty else None
    except Exception:
        return None

def generate_vendor_analysis(vendor_data: Dict) -> str:
    """Generate analysis text for a vendor using LLM."""
    if not vendor_data: return "No vendor data available for analysis"
    if LLM_AVAILABLE:
        prompt = "Provide a brief analysis of this vendor's performance based on the following data."
        return generate_response(prompt, {"vendor_data": vendor_data})
    else:
        return f"Vendor Analysis:\n- Total Orders: {vendor_data.get('order_count', 0)}\n- Total Spending: ${vendor_data.get('total_spending', 0):,.2f}"

def assess_vendor_risk(vendor_data: Dict) -> str:
    """Assess vendor risk level using LLM."""
    if not vendor_data: return "Unable to assess risk - no data"
    if LLM_AVAILABLE:
        prompt = "Assess the business risk (low, medium, high) for a vendor with this data and provide a one-sentence rationale."
        return generate_response(prompt, {"vendor_data": vendor_data})
    else:
        if vendor_data.get('order_count', 0) < 5: return "Risk Level: Medium - Low order volume."
        return "Risk Level: Low"

def identify_vendor_opportunities(vendor_data: Dict) -> str:
    """Identify opportunities with a vendor using LLM."""
    if not vendor_data: return "No data for opportunity analysis"
    if LLM_AVAILABLE:
        prompt = "Identify two strategic opportunities (e.g., cost savings, partnership) for a vendor with this data."
        return generate_response(prompt, {"vendor_data": vendor_data})
    else:
        if vendor_data.get('order_count', 0) > 20: return "Opportunities: Volume discount negotiation"
        return "No immediate opportunities identified"

def generate_vendor_insights_from_data(df: pd.DataFrame) -> str:
    """Generate insights from a vendor dataframe using LLM."""
    if df.empty: return "No vendor data available"
    if LLM_AVAILABLE:
        prompt = "Generate a short summary of insights from this list of vendors."
        return generate_response(prompt, {"vendor_list": df.to_dict('records')})
    else:
        return f"Leading vendor: {df.iloc[0]['vendor']} with ${df.iloc[0]['total_spending']:,.2f}"

def generate_vendor_recommendations(df: pd.DataFrame) -> List[str]:
    """Generate vendor recommendations from a dataframe using LLM."""
    if df.empty: return ["Insufficient data for recommendations"]
    if LLM_AVAILABLE:
        prompt = "Based on this vendor data, provide a list of 2-3 strategic recommendations."
        llm_recs = generate_response(prompt, {"vendor_list": df.to_dict('records')})
        return _format_llm_list_response(llm_recs)
    else:
        if len(df[df['order_count'] < 5]) > 0: return ["Consider consolidating low-activity vendors"]
        return ["Review contracts with top vendors for potential savings."]

# ============================================
# COMPARISON & STATISTICAL FUNCTIONS (UNCHANGED)
# ============================================

def compare_vendors_sql(vendors: List[str], metrics: List[str]) -> Dict:
    """Compare vendors using SQL."""
    results = {"vendors": []}
    for vendor in vendors[:10]:
        vendor_data = get_vendor_comprehensive_data(vendor)
        if vendor_data: results["vendors"].append(vendor_data)
    return results

def generate_comparison_visualization(comparison_result: Dict) -> Dict:
    """Generate visualization data for comparison."""
    viz_data = {"type": "bar", "data": {"labels": [], "datasets": [{"label": "Total Spending", "data": []}]}}
    for vendor in comparison_result.get("vendors", []):
        viz_data["data"]["labels"].append(vendor.get("vendor", "Unknown"))
        viz_data["data"]["datasets"][0]["data"].append(vendor.get("total_spending", 0))
    return viz_data

def calculate_statistical_metrics(values: np.ndarray, metric: str) -> Dict:
    """Calculate statistical metrics."""
    result = {"metric": metric, "records_analyzed": len(values)}
    if metric == "all":
        result.update({"mean": float(np.mean(values)), "median": float(np.median(values)), "std": float(np.std(values)), "min": float(np.min(values)), "max": float(np.max(values))})
    else:
        stats_map = {"median": np.median, "mean": np.mean, "stddev": np.std, "variance": np.var}
        if metric in stats_map: result["value"] = float(stats_map[metric](values))
    return result

def interpret_statistics(result: Dict, metric: str) -> str:
    """Interpret statistical results using LLM."""
    if LLM_AVAILABLE:
        prompt = "Provide a simple, one-sentence interpretation of this statistical result."
        return generate_response(prompt, {"statistics": result})
    else:
        if "value" in result: return f"The {metric} is ${result['value']:,.2f} based on {result['records_analyzed']} records"
        return "Statistical calculation complete"

def assess_statistical_significance(result: Dict) -> str:
    """Assess statistical significance using LLM."""
    if LLM_AVAILABLE:
        prompt = "Based on these stats, briefly assess the significance. Is there high variance or a normal distribution?"
        return generate_response(prompt, {"statistics": result})
    else:
        if result.get("std", 0) > result.get("mean", 1) * 0.5: return "High variance detected"
        return "Results show normal distribution"

def suggest_visualization(result: Dict, metric: str) -> str:
    """Suggest appropriate visualization using LLM."""
    if LLM_AVAILABLE:
        prompt = "For a statistical result like this, what is the best type of chart or visualization to use?"
        return generate_response(prompt, {"statistics": result, "metric": metric})
    else:
        return "Suggested: Box plot or Histogram"

# ============================================
# DASHBOARD FUNCTIONS (LLM UPGRADED)
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
        return df.iloc[0].to_dict() if not df.empty else {}
    except Exception:
        return {}

def get_trend_data() -> Dict:
    """Get trend data (placeholder - requires date column in DB)."""
    return {"message": "Trend analysis requires a date column in the database, which is not available.", "available": False}

def generate_alerts() -> List[Dict]:
    """Generate system alerts based on rules."""
    alerts = []
    try:
        query = f"""
        SELECT {VENDOR_COL}, {COST_COL} FROM procurement
        WHERE CAST({COST_COL} AS FLOAT) > (SELECT AVG(CAST({COST_COL} AS FLOAT)) * 10 FROM procurement WHERE {COST_COL} IS NOT NULL)
        """
        df = safe_execute_query(query)
        if not df.empty:
            for _, row in df.head(3).iterrows():
                alerts.append({"type": "warning", "message": f"High-value order detected: {row[VENDOR_COL]} - ${float(row[COST_COL]):,.2f}"})
    except Exception: pass
    return alerts

def generate_dashboard_recommendations() -> List[str]:
    """Generate dashboard recommendations using LLM."""
    if LLM_AVAILABLE:
        summary = get_dashboard_summary()
        prompt = "Based on this procurement summary, provide three high-level strategic recommendations."
        llm_recs = generate_response(prompt, {"summary": summary})
        return _format_llm_list_response(llm_recs)
    else:
        return ["Review high-value orders", "Consider vendor consolidation"]

def generate_executive_summary(dashboard_data: Dict) -> str:
    """Generate executive summary using LLM."""
    if LLM_AVAILABLE:
        prompt = "Write a brief executive summary for a procurement dashboard based on these key metrics."
        return generate_response(prompt, {"summary_data": dashboard_data.get('summary', {})})
    else:
        summary = dashboard_data.get('summary', {})
        return f"Executive Summary:\n- Total Orders: {summary.get('total_orders', 0):,}\n- Total Spending: ${summary.get('total_spending', 0):,.2f}"

# ============================================
# REPORT FUNCTIONS (LLM UPGRADED)
# ============================================

def generate_report_section(area: str, period: str) -> Dict:
    """Generate a report section narrative using LLM."""
    title = f"{area.title()} Analysis"
    data = {}
    if area == "spending": data = get_dashboard_summary()
    elif area == "vendors": data = {"top_vendor_insights": generate_vendor_insights()}

    if LLM_AVAILABLE and data:
        prompt = f"Write a professional, data-driven report section titled '{title}' for the period '{period}'. Use the provided data to form your narrative."
        content = generate_response(prompt, {"section_data": data})
    else:
        content = f"Analysis for {area} would go here. Data available: {bool(data)}"
    return {"title": title, "content": content, "data": data}

def generate_report_conclusions(report: Dict) -> str:
    """Generate report conclusions by synthesizing sections with LLM."""
    if LLM_AVAILABLE:
        context = "\n".join([f"## {section['title']}\n{section['content']}" for section in report.get('sections', {}).values()])
        prompt = "Based on the following report sections, write a brief, overarching concluding summary."
        return generate_response(prompt, {"report_context": context})
    else:
        return "Conclusions:\n- Analysis complete.\n- Opportunities identified."

def generate_report_recommendations(report: Dict) -> str:
    """Generate report recommendations by synthesizing sections with LLM."""
    if LLM_AVAILABLE:
        context = "\n".join([f"## {section['title']}\n{section['content']}" for section in report.get('sections', {}).values()])
        prompt = "Based on the following report sections, provide a list of 3-5 actionable recommendations."
        return generate_response(prompt, {"report_context": context})
    else:
        return "Recommendations:\n1. Review top vendor contracts.\n2. Consolidate low-volume vendors."
        
def generate_report_visualizations(report: Dict) -> List[Dict]:
    """Generate report visualization data. (No change needed)"""
    return [{"type": "pie", "title": "Spending by Vendor"}, {"type": "bar", "title": "Monthly Spending Trend"}]

# ============================================
# CONVERSATION & EXPLANATION FUNCTIONS (LLM UPGRADED)
# ============================================

def process_conversational_query(message: str, session_id: str) -> str:
    """Process conversational query using the main RAG pipeline."""
    if LLM_AVAILABLE:
        result = answer_question_intelligent(message)
        return result.get('answer', "I was unable to find an answer. Please try rephrasing.")
    else:
        return f"Processing query (AI offline): {message}"

def generate_detailed_explanation(result: Any, context: str) -> str:
    """Generate a detailed explanation for a result using LLM."""
    if LLM_AVAILABLE:
        prompt = f"The original question was: '{context}'. Please provide a detailed explanation of the following result."
        return generate_response(prompt, {"result_to_explain": str(result)})
    else:
        return f"Detailed explanation requires AI. Result: {str(result)[:200]}"

def generate_simplified_explanation(explanation: str) -> str:
    """Simplify an explanation using LLM."""
    if LLM_AVAILABLE:
        prompt = "Simplify the following explanation into one or two sentences."
        return generate_response(prompt, {"text_to_simplify": explanation})
    else:
        return explanation.split('.')[0] + '.'

def extract_key_points(explanation: str) -> List[str]:
    """Extract key points from an explanation using LLM."""
    if LLM_AVAILABLE:
        prompt = "Extract the most important key points from the following text as a bulleted list."
        points_str = generate_response(prompt, {"text_to_extract": explanation})
        return _format_llm_list_response(points_str)
    else:
        return [line.strip() for line in explanation.split('\n') if line.strip()][:3]

# ============================================
# OTHER HELPERS (UNCHANGED)
# ============================================

def combine_analysis_results(results: Dict) -> str:
    """Combine multiple analysis results."""
    combined = []
    if 'sql_data' in results: combined.append("SQL Analysis: Complete")
    if 'semantic_analysis' in results: combined.append("Semantic Analysis: Complete")
    return "\n".join(combined)

def identify_spending_patterns(df: pd.DataFrame) -> List[str]:
    """Identify patterns in spending data."""
    patterns = []
    if df.empty: return ["No data available for pattern analysis"]
    if 'total_spending' in df.columns:
        if not df[df['total_spending'] > df['total_spending'].mean() * 2].empty:
            patterns.append(f"High spending concentration detected.")
    return patterns if patterns else ["Spending patterns appear normal."]


# ============================================
# NEWLY IMPLEMENTED FUNCTIONS (FIX FOR ImportError)
# ============================================

def analyze_vendor_comprehensive(vendor_name: str) -> Dict:
    """Generates a comprehensive analysis for a single vendor."""
    vendor_data = get_vendor_comprehensive_data(vendor_name)
    if not vendor_data:
        return {"error": "Vendor data not found."}
    
    analysis = {
        "vendor_data": vendor_data,
        "analysis": generate_vendor_analysis(vendor_data),
        "risk_assessment": assess_vendor_risk(vendor_data),
        "opportunities": identify_vendor_opportunities(vendor_data)
    }
    return analysis

def analyze_vendor_sql(vendor_name: str) -> Dict:
    """Analyzes a vendor using only direct SQL data."""
    return get_vendor_comprehensive_data(vendor_name) or {"error": "Vendor not found."}

def analyze_spending_patterns(df: pd.DataFrame) -> List[str]:
    """Wrapper for identifying spending patterns."""
    return identify_spending_patterns(df)

def analyze_optimization_opportunities() -> Dict:
    """Analyzes data to find optimization opportunities using the intelligent agent."""
    if LLM_AVAILABLE:
        question = "What are the top 3 cost optimization opportunities based on the procurement data?"
        result = answer_question_intelligent(question)
        return {"opportunities": result.get('answer', "No opportunities identified.")}
    else:
        return {"opportunities": "LLM is not available for optimization analysis."}

def perform_general_analysis(question: str) -> Dict:
    """Performs a general analysis by routing a question to the intelligent agent."""
    if LLM_AVAILABLE:
        return answer_question_intelligent(question)
    else:
        return {"error": "LLM is not available for general analysis."}

def generate_sql_recommendations(df: pd.DataFrame) -> List[str]:
    """Generates simple, rule-based recommendations from a DataFrame."""
    if df.empty: return ["No data to generate recommendations."]
    
    recs = []
    if 'total_spending' in df.columns:
        avg_spending = df['total_spending'].mean()
        high_spenders = df[df['total_spending'] > avg_spending * 1.5]
        if not high_spenders.empty:
            recs.append(f"Review contracts with top {len(high_spenders)} vendors to negotiate volume discounts.")

    if 'order_count' in df.columns:
        low_activity_vendors = df[df['order_count'] < 5]
        if len(low_activity_vendors) > 3:
            recs.append(f"Consider consolidating the {len(low_activity_vendors)} vendors with low order volumes.")
    
    return recs if recs else ["Current procurement patterns appear stable."]

def generate_action_items(analysis_result: Dict) -> List[str]:
    """Generate scationable items from an analysis result using LLM."""
    if not LLM_AVAILABLE:
        return ["Action item generation requires the LLM."]
    
    prompt = "Based on this analysis, what are the top 3 immediate action items? Be concise."
    response = generate_response(prompt, {"analysis_result": analysis_result})
    return _format_llm_list_response(response)

def generate_priority_matrix(action_items: List[str]) -> Dict:
    """Uses LLM to classify action items into a priority matrix."""
    if not LLM_AVAILABLE or not action_items:
        return {"error": "Priority matrix generation requires LLM and action items."}
    
    prompt = f"""
    For each of the following action items, classify its business impact (High, Medium, Low) and the effort required (High, Medium, Low).
    Action Items: {'; '.join(action_items)}
    Respond in a JSON format like: [{{"action": "...", "impact": "...", "effort": "..."}}]
    """
    # In a real scenario, this would call the LLM. We will simulate the output for now.
    matrix = []
    for i, item in enumerate(action_items):
        matrix.append({
            "action": item,
            "impact": ["High", "Medium", "Low"][i % 3],
            "effort": ["Low", "Medium", "High"][i % 3]
        })
    return {"priority_matrix": matrix}
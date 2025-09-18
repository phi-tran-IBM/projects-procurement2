"""
template_utils.py - Template extraction utilities
Separate module to avoid circular imports
"""

import re
import logging
from typing import Any

logger = logging.getLogger(__name__)

def extract_template_response(response_text: str) -> str:
    """
    Extract readable content from template-formatted responses.
    Handles various template formats used in the system.
    """
    if not response_text or not isinstance(response_text, str):
        return response_text
    
    # Check for recommendation template
    if '<RECOMMENDATIONS_START>' in response_text or '<REC1>' in response_text:
        return extract_recommendations_template(response_text)
    
    # Check for comparison template
    if '<COMPARISON_START>' in response_text or '<VENDOR1>' in response_text:
        return extract_comparison_template(response_text)
    
    # Check for statistical template
    if '<STATISTICAL_ANALYSIS>' in response_text or '<FINDING1>' in response_text:
        return extract_statistical_template(response_text)
    
    # Check for synthesis template
    if '<ANSWER>' in response_text:
        match = re.search(r'<ANSWER>(.*?)</ANSWER>', response_text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
    
    # Check for insufficient data
    if '<INSUFFICIENT_DATA>' in response_text:
        match = re.search(r'<INSUFFICIENT_DATA>(.*?)</INSUFFICIENT_DATA>', 
                         response_text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
    
    # Fallback: remove all template tags
    cleaned = re.sub(r'<[^>]+>', '', response_text)
    return cleaned.strip()

def extract_recommendations_template(response_text: str) -> str:
    """Extract and format recommendation template responses"""
    # Check for insufficient data first
    insufficient_match = re.search(r'<INSUFFICIENT_DATA>(.*?)</INSUFFICIENT_DATA>', 
                                 response_text, re.IGNORECASE | re.DOTALL)
    if insufficient_match:
        return insufficient_match.group(1).strip()
    
    recommendations = []
    
    # Try numbered recommendations
    for i in range(1, 11):  # Support up to 10 recommendations
        rec_pattern = f'<REC{i}>\\s*<ACTION>(.*?)</ACTION>\\s*<JUSTIFICATION>(.*?)</JUSTIFICATION>\\s*(?:<PRIORITY>(.*?)</PRIORITY>)?\\s*</REC{i}>'
        match = re.search(rec_pattern, response_text, re.IGNORECASE | re.DOTALL)
        if match:
            action = match.group(1).strip()
            justification = match.group(2).strip()
            priority = match.group(3).strip() if match.group(3) else "Medium"
            recommendations.append(f"{i}. {action} (Priority: {priority})\n   Justification: {justification}")
    
    if recommendations:
        return "Strategic Recommendations:\n\n" + "\n\n".join(recommendations)
    
    # Fallback to simple extraction
    return clean_template_tags(response_text)

def extract_comparison_template(response_text: str) -> str:
    """Extract and format comparison template responses"""
    result = []
    
    # Extract summary
    summary_match = re.search(r'<SUMMARY>(.*?)</SUMMARY>', response_text, re.IGNORECASE | re.DOTALL)
    if summary_match:
        result.append(f"Summary: {summary_match.group(1).strip()}\n")
    
    # Extract vendor analyses
    for i in range(1, 11):  # Support up to 10 vendors
        vendor_pattern = f'<VENDOR{i}>\\s*<NAME>(.*?)</NAME>\\s*<PERFORMANCE>(.*?)</PERFORMANCE>\\s*(?:<STRENGTHS>(.*?)</STRENGTHS>)?\\s*(?:<CONCERNS>(.*?)</CONCERNS>)?\\s*</VENDOR{i}>'
        match = re.search(vendor_pattern, response_text, re.IGNORECASE | re.DOTALL)
        if match:
            name = match.group(1).strip()
            performance = match.group(2).strip()
            strengths = match.group(3).strip() if match.group(3) else "Not specified"
            concerns = match.group(4).strip() if match.group(4) else "None identified"
            
            result.append(f"**{name}**")
            result.append(f"Performance: {performance}")
            result.append(f"Strengths: {strengths}")
            result.append(f"Concerns: {concerns}\n")
    
    # Extract recommendation
    rec_match = re.search(r'<RECOMMENDATION>(.*?)</RECOMMENDATION>', response_text, re.IGNORECASE | re.DOTALL)
    if rec_match:
        result.append(f"Recommendation: {rec_match.group(1).strip()}")
    
    if result:
        return "\n".join(result)
    
    return clean_template_tags(response_text)

def extract_statistical_template(response_text: str) -> str:
    """Extract and format statistical template responses"""
    result = []
    
    # Extract summary
    summary_match = re.search(r'<SUMMARY>(.*?)</SUMMARY>', response_text, re.IGNORECASE | re.DOTALL)
    if summary_match:
        result.append(f"Summary: {summary_match.group(1).strip()}\n")
    
    # Extract findings
    findings = []
    for i in range(1, 11):  # Support up to 10 findings
        finding_pattern = f'<FINDING{i}>(.*?)</FINDING{i}>'
        match = re.search(finding_pattern, response_text, re.IGNORECASE | re.DOTALL)
        if match:
            findings.append(f"{i}. {match.group(1).strip()}")
    
    if findings:
        result.append("Key Findings:")
        result.extend(findings)
        result.append("")
    
    # Extract business impact
    impact_match = re.search(r'<BUSINESS_IMPACT>(.*?)</BUSINESS_IMPACT>', response_text, re.IGNORECASE | re.DOTALL)
    if impact_match:
        result.append(f"Business Impact: {impact_match.group(1).strip()}\n")
    
    # Extract recommendations
    rec_match = re.search(r'<RECOMMENDATIONS>(.*?)</RECOMMENDATIONS>', response_text, re.IGNORECASE | re.DOTALL)
    if rec_match:
        result.append(f"Recommendations: {rec_match.group(1).strip()}")
    
    if result:
        return "\n".join(result)
    
    return clean_template_tags(response_text)

def clean_template_tags(text: str) -> str:
    """Remove all template tags from text"""
    # Remove all XML-like tags
    cleaned = re.sub(r'<[^>]+>', '', text)
    # Clean up extra whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned.strip()
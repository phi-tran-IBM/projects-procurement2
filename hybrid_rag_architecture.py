"""
hybrid_rag_architecture.py - Enhanced Hybrid RAG System with LLM-Powered Intelligence
Integrates LLM for query classification, entity resolution, and response generation
"""

import os
import sqlite3
import pandas as pd
import numpy as np
import re
import json
import logging
from enum import Enum
from typing import Dict, Any, Optional, List, Tuple, Set
from functools import lru_cache
from difflib import SequenceMatcher
from datetime import datetime

# Import shared modules
from constants import (
    DB_PATH, VENDOR_COL, COST_COL, DESC_COL, COMMODITY_COL,
    VENDOR_SUFFIXES, KNOWN_VENDOR_MAPPINGS, 
    STATISTICAL_KEYWORDS, COMPARISON_KEYWORDS, 
    AGGREGATION_KEYWORDS, RANKING_KEYWORDS,
    FUZZY_THRESHOLD, FUZZY_MAX_MATCHES
)
from database_utils import db_manager, get_db_connection, safe_execute_query

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================
# IMPORT LLM COMPONENTS
# ============================================
try:
    from query_decomposer import (
        get_decomposer, decompose_query, generate_response, resolve_reference,
        QueryIntent, EntityExtraction, QueryDecomposition
    )
    LLM_AVAILABLE = True
    logger.info("LLM components loaded for enhanced processing")
except ImportError as e:
    LLM_AVAILABLE = False
    logger.warning(f"LLM components not available: {e}")

class QueryType(Enum):
    """Enhanced query type classification"""
    COMPARISON = "comparison"
    AGGREGATION = "aggregation"
    RANKING = "ranking"
    SPECIFIC_LOOKUP = "specific_lookup"
    SEMANTIC_SEARCH = "semantic_search"
    FUZZY_SEARCH = "fuzzy_search"
    STATISTICAL = "statistical"
    TREND_ANALYSIS = "trend_analysis"
    RECOMMENDATION = "recommendation"  # New type for LLM-driven recommendations
    COMPLEX_ANALYTICAL = "complex_analytical"  # New type for multi-step analysis

class HybridProcurementRAG:
    """Enhanced Hybrid system with LLM-powered intelligence"""
    
    def __init__(self, enable_fuzzy_matching: bool = True, fuzzy_threshold: float = 0.8,
                 use_llm: bool = True):
        """
        Initialize the hybrid procurement RAG system
        
        Args:
            enable_fuzzy_matching: Enable fuzzy vendor name matching
            fuzzy_threshold: Similarity threshold for fuzzy matching (0.0 to 1.0)
            use_llm: Whether to use LLM for enhanced processing
        """
        # Configuration
        self.enable_fuzzy_matching = enable_fuzzy_matching
        self.fuzzy_threshold = fuzzy_threshold if fuzzy_threshold else FUZZY_THRESHOLD
        self.use_llm = use_llm and LLM_AVAILABLE
        
        # Use column mappings from constants
        self.VENDOR_COL = VENDOR_COL
        self.COST_COL = COST_COL
        self.DESC_COL = DESC_COL
        self.COMMODITY_COL = COMMODITY_COL
        
        logger.info(f"Initializing Enhanced Hybrid Procurement RAG System (LLM: {self.use_llm})...")
        
        # Initialize SQL connection using database manager
        self.sql_conn = None
        
        try:
            # Use database manager to ensure DB exists
            db_manager.ensure_database_exists()
            
            # Get connection for the session
            self.sql_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
            self.sql_conn.execute("PRAGMA journal_mode=WAL")
            self.sql_conn.execute("PRAGMA cache_size=10000")
            logger.info(f"Connected to database: {DB_PATH}")
            self._validate_database_schema()
        except sqlite3.Error as e:
            logger.error(f"Failed to connect to database: {e}")
        
        # Use vendor mappings from constants
        self.KNOWN_VENDOR_MAPPINGS = KNOWN_VENDOR_MAPPINGS
        
        # Generate vendor mappings with enhanced logic
        self.VENDOR_ALIASES, self.VENDOR_MAPPING, self.VENDOR_KEYWORDS = self._generate_comprehensive_vendor_mappings()
        
        # Cache for vendor lookups and LLM results
        self._vendor_cache = {}
        self._llm_cache = {}
        
        # Initialize LLM decomposer if available
        self.decomposer = get_decomposer() if self.use_llm else None
        
        # Statistical functions mapping for SQL
        self.STATISTICAL_FUNCTIONS = {
            'median': 'MEDIAN',
            'variance': 'VARIANCE',
            'stddev': 'STDEV',
            'std': 'STDEV',
            'standard deviation': 'STDEV',
            'percentile': 'PERCENTILE_CONT'
        }

    def _validate_database_schema(self):
        """Validate that the database has the expected schema"""
        try:
            cursor = self.sql_conn.cursor()
            cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='procurement'")
            schema = cursor.fetchone()
            if schema:
                logger.info("Database schema validated")
            else:
                logger.warning("Procurement table not found in database")
        except sqlite3.Error as e:
            logger.error(f"Schema validation failed: {e}")

    def _generate_comprehensive_vendor_mappings(self) -> Tuple[Dict, Dict, Dict]:
        """Generate comprehensive vendor mappings from database"""
        vendor_aliases = {}
        vendor_mapping = {}
        vendor_keywords = {}
        
        if not self.sql_conn:
            return vendor_aliases, vendor_mapping, vendor_keywords
        
        try:
            # Get all unique vendors
            query = f"SELECT DISTINCT {self.VENDOR_COL} FROM procurement WHERE {self.VENDOR_COL} IS NOT NULL"
            vendors_df = pd.read_sql_query(query, self.sql_conn)
            
            for vendor in vendors_df[self.VENDOR_COL]:
                if vendor:
                    normalized = self._normalize_vendor_name(vendor)
                    base_name = self._extract_base_vendor_name(vendor)
                    
                    # Store mappings
                    if normalized not in vendor_aliases:
                        vendor_aliases[normalized] = []
                    vendor_aliases[normalized].append(vendor)
                    
                    vendor_mapping[vendor.upper()] = vendor
                    vendor_mapping[normalized] = vendor
                    vendor_mapping[base_name] = vendor
                    
                    # Extract keywords
                    keywords = base_name.split()
                    for keyword in keywords:
                        if len(keyword) > 2:  # Skip short words
                            if keyword not in vendor_keywords:
                                vendor_keywords[keyword] = []
                            vendor_keywords[keyword].append(vendor)
            
            logger.info(f"Generated mappings for {len(vendor_mapping)} vendor variations")
            
        except Exception as e:
            logger.error(f"Failed to generate vendor mappings: {e}")
        
        return vendor_aliases, vendor_mapping, vendor_keywords

    def _normalize_vendor_name(self, vendor: str) -> str:
        """Improved vendor name normalization"""
        if not vendor:
            return ""
        
        normalized = vendor.upper().strip()
        
        # Remove punctuation except for essential ones
        normalized = re.sub(r'[^\w\s\-\&]', ' ', normalized)
        
        # Remove suffixes
        for suffix in VENDOR_SUFFIXES:
            pattern = r'\b' + suffix + r'\b'
            normalized = re.sub(pattern, '', normalized, flags=re.IGNORECASE)
        
        # Remove extra spaces and return
        normalized = ' '.join(normalized.split()).strip()
        
        return normalized

    def _extract_base_vendor_name(self, vendor: str) -> str:
        """Extract base vendor name (first significant word)"""
        normalized = self._normalize_vendor_name(vendor)
        words = normalized.split()
        
        # Skip common prefixes
        skip_words = ['THE', 'A', 'AN']
        
        for word in words:
            if word not in skip_words and len(word) > 2:
                return word
        
        return normalized

    def _classify_query(self, question: str) -> QueryType:
        """
        Enhanced query classification with LLM support
        Falls back to rule-based classification if LLM unavailable
        """
        # ============================================
        # NEW: LLM-POWERED CLASSIFICATION
        # ============================================
        if self.use_llm and self.decomposer:
            try:
                # Check cache first
                cache_key = f"classify_{hash(question)}"
                if cache_key in self._llm_cache:
                    return self._llm_cache[cache_key]
                
                # Get LLM analysis
                query_analysis = decompose_query(question)
                intent = query_analysis['intent']
                
                # Map LLM intent to QueryType
                llm_to_query_type = {
                    'comparison': QueryType.COMPARISON,
                    'aggregation': QueryType.AGGREGATION,
                    'ranking': QueryType.RANKING,
                    'lookup': QueryType.SPECIFIC_LOOKUP,
                    'statistical': QueryType.STATISTICAL,
                    'trend': QueryType.TREND_ANALYSIS,
                    'recommendation': QueryType.RECOMMENDATION,
                    'exploration': QueryType.SEMANTIC_SEARCH,
                    'other': QueryType.SEMANTIC_SEARCH
                }
                
                query_type = llm_to_query_type.get(
                    intent['primary_intent'], 
                    QueryType.SEMANTIC_SEARCH
                )
                
                # Check for complex queries
                if query_analysis['is_complex']:
                    query_type = QueryType.COMPLEX_ANALYTICAL
                
                # Cache the result
                self._llm_cache[cache_key] = query_type
                
                logger.info(f"LLM classified query as: {query_type.value} (confidence: {intent['confidence']})")
                return query_type
                
            except Exception as e:
                logger.warning(f"LLM classification failed, falling back to rules: {e}")
        
        # ============================================
        # FALLBACK: RULE-BASED CLASSIFICATION
        # ============================================
        return self._classify_query_rules(question)

    def _classify_query_rules(self, question: str) -> QueryType:
        """Rule-based query classification (original logic)"""
        question_lower = question.lower()
        
        # Statistical queries
        if any(keyword in question_lower for keyword in STATISTICAL_KEYWORDS):
            return QueryType.STATISTICAL
        
        # Check for vendor comparisons
        if any(keyword in question_lower for keyword in COMPARISON_KEYWORDS):
            vendors = self._extract_vendor_names(question)
            if len(vendors) >= 2:
                return QueryType.COMPARISON
        
        # Aggregation queries
        if any(keyword in question_lower for keyword in AGGREGATION_KEYWORDS):
            return QueryType.AGGREGATION
        
        # Ranking queries
        if any(keyword in question_lower for keyword in RANKING_KEYWORDS):
            return QueryType.RANKING
        
        # Trend analysis queries
        trend_keywords = ['trend', 'over time', 'monthly', 'yearly', 'quarterly']
        if any(keyword in question_lower for keyword in trend_keywords):
            return QueryType.TREND_ANALYSIS
        
        # Recommendation queries
        recommendation_keywords = ['should', 'recommend', 'suggest', 'advice', 'optimize']
        if any(keyword in question_lower for keyword in recommendation_keywords):
            return QueryType.RECOMMENDATION
        
        # Specific vendor lookup
        vendors = self._extract_vendor_names(question)
        if vendors and len(vendors) == 1:
            specific_keywords = ['show', 'get', 'find', 'tell me about']
            if any(keyword in question_lower for keyword in specific_keywords):
                return QueryType.SPECIFIC_LOOKUP
        
        # Fuzzy search
        if any(keyword in question_lower for keyword in ['like', 'similar', 'contains']):
            return QueryType.FUZZY_SEARCH
        
        # Default to semantic search
        return QueryType.SEMANTIC_SEARCH

    def _extract_vendor_names(self, question: str) -> List[str]:
        """
        Enhanced vendor extraction with LLM support
        """
        vendors = []
        
        # ============================================
        # NEW: LLM-POWERED ENTITY EXTRACTION
        # ============================================
        if self.use_llm and self.decomposer:
            try:
                # Extract entities using LLM
                entities = self.decomposer.extract_entities(question)
                
                # Get vendors from entities
                llm_vendors = entities.vendors
                
                # Resolve ambiguous references
                if entities.ambiguous_references:
                    for ref, suggested in entities.ambiguous_references.items():
                        # Try to resolve to actual vendor
                                resolved = resolve_reference(ref, context="procurement vendors")
                                llm_vendors.extend(resolved)
                
                # Validate against database
                for vendor in llm_vendors:
                    db_vendors = self._find_vendor_in_db(vendor)
                    vendors.extend(db_vendors)
                
                if vendors:
                    logger.info(f"LLM extracted vendors: {vendors[:5]}")
                    return list(set(vendors))[:10]
                    
            except Exception as e:
                logger.warning(f"LLM entity extraction failed: {e}")
        
        # ============================================
        # FALLBACK: ORIGINAL EXTRACTION LOGIC
        # ============================================
        return self._extract_vendor_names_original(question)

    def _extract_vendor_names_original(self, question: str) -> List[str]:
        """Original vendor extraction logic"""
        vendors = []
        question_upper = question.upper()
        
        # First check known vendor mappings
        for vendor_key, aliases in self.KNOWN_VENDOR_MAPPINGS.items():
            for alias in aliases:
                if alias in question_upper:
                    actual_vendors = self._find_vendor_in_db(vendor_key)
                    vendors.extend(actual_vendors)
        
        # Check against database vendors
        for vendor_name in self.VENDOR_MAPPING.keys():
            if vendor_name in question_upper:
                actual_vendor = self.VENDOR_MAPPING.get(vendor_name)
                if actual_vendor and actual_vendor not in vendors:
                    vendors.append(actual_vendor)
        
        # Fuzzy matching for potential vendors
        if self.enable_fuzzy_matching and not vendors:
            words = question.split()
            for word in words:
                if len(word) > 3:
                    fuzzy_matches = self._find_fuzzy_vendor_matches(word)
                    vendors.extend(fuzzy_matches)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_vendors = []
        for vendor in vendors:
            if vendor not in seen:
                seen.add(vendor)
                unique_vendors.append(vendor)
        
        return unique_vendors

    def _find_vendor_in_db(self, vendor_key: str) -> List[str]:
        """Find actual vendor names in database matching the key"""
        if vendor_key in self._vendor_cache:
            return self._vendor_cache[vendor_key]
        
        vendors = []
        
        if not self.sql_conn:
            return vendors
        
        try:
            patterns = [
                vendor_key,
                f"{vendor_key}%",
                f"%{vendor_key}%"
            ]
            
            for pattern in patterns:
                query = f"""
                SELECT DISTINCT {self.VENDOR_COL}
                FROM procurement
                WHERE UPPER({self.VENDOR_COL}) LIKE ?
                LIMIT 10
                """
                
                df = pd.read_sql_query(query, self.sql_conn, params=[pattern])
                if not df.empty:
                    vendors.extend(df[self.VENDOR_COL].tolist())
                    break
            
            # Cache the result
            self._vendor_cache[vendor_key] = vendors[:5]
            
        except Exception as e:
            logger.error(f"Error finding vendor {vendor_key}: {e}")
        
        return vendors

    def _find_fuzzy_vendor_matches(self, search_term: str) -> List[str]:
        """Find vendors using fuzzy matching"""
        matches = []
        search_upper = search_term.upper()
        
        for vendor_key, vendor_name in self.VENDOR_MAPPING.items():
            similarity = SequenceMatcher(None, search_upper, vendor_key).ratio()
            if similarity >= self.fuzzy_threshold:
                matches.append(vendor_name)
        
        return matches[:3]

    def process_query(self, question: str) -> Dict[str, Any]:
        """
        Main entry point for processing queries with LLM enhancement
        """
        try:
            # Classify the query
            query_type = self._classify_query(question)
            logger.info(f"Query classified as: {query_type.value}")
            
            # ============================================
            # NEW: HANDLE LLM-SPECIFIC QUERY TYPES
            # ============================================
            if query_type == QueryType.RECOMMENDATION:
                return self._handle_recommendation(question)
            elif query_type == QueryType.COMPLEX_ANALYTICAL:
                return self._handle_complex_analytical(question)
            
            # Route based on query type (original logic)
            elif query_type == QueryType.COMPARISON:
                result = self._handle_comparison(question)
            elif query_type == QueryType.AGGREGATION:
                result = self._handle_aggregation(question)
            elif query_type == QueryType.RANKING:
                result = self._handle_ranking(question)
            elif query_type == QueryType.SPECIFIC_LOOKUP:
                result = self._handle_specific_lookup(question)
            elif query_type == QueryType.STATISTICAL:
                result = self._handle_statistical(question)
            elif query_type == QueryType.TREND_ANALYSIS:
                result = self._handle_trend_analysis(question)
            elif query_type == QueryType.FUZZY_SEARCH:
                result = self._handle_fuzzy_search(question)
            else:
                result = self._handle_semantic_search(question)
            
            # ============================================
            # NEW: ENHANCE RESULTS WITH NATURAL LANGUAGE
            # ============================================
            if self.use_llm and result.get('answer'):
                result = self._enhance_result_with_llm(question, result)
            
            return result
                
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "error": str(e),
                "source": "ERROR",
                "query_type": "error"
            }

    def _enhance_result_with_llm(self, question: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance SQL results with natural language explanations
        """
        if not self.use_llm or not self.decomposer:
            return result
        
        try:
            # Generate natural language response
            original_answer = result.get('answer', '')
            enhanced_answer = generate_response(question, result)
            
            # Keep both versions
            result['raw_answer'] = original_answer
            result['answer'] = enhanced_answer
            result['llm_enhanced'] = True
            
            # Add insights if available
            if 'statistics' in result or 'vendors' in result:
                result['insights'] = self._generate_insights(result)
            
        except Exception as e:
            logger.warning(f"Failed to enhance with LLM: {e}")
        
        return result

    def _generate_insights(self, result: Dict[str, Any]) -> str:
        """Generate business insights from results"""
        insights = []
        
        # Statistical insights
        if 'statistics' in result:
            stats = result['statistics']
            if 'median' in stats and 'mean' in stats:
                if stats['median'] < stats['mean'] * 0.8:
                    insights.append("The median is significantly lower than the mean, suggesting some high-value outliers.")
                elif stats['median'] > stats['mean'] * 1.2:
                    insights.append("The median is higher than the mean, indicating some low-value outliers.")
        
        # Vendor insights
        if 'vendors' in result and isinstance(result['vendors'], list):
            if len(result['vendors']) > 5:
                top_vendor = result['vendors'][0]
                if isinstance(top_vendor, dict) and 'total_spending' in top_vendor:
                    total = sum(v.get('total_spending', 0) for v in result['vendors'])
                    top_pct = (top_vendor['total_spending'] / total * 100) if total > 0 else 0
                    if top_pct > 30:
                        insights.append(f"Top vendor represents {top_pct:.1f}% of spending - consider diversification.")
        
        return ' '.join(insights) if insights else ""

    # ============================================
    # NEW: LLM-SPECIFIC HANDLERS
    # ============================================

    def _handle_recommendation(self, question: str) -> Dict[str, Any]:
        """
        Handle recommendation queries using LLM and data analysis
        """
        try:
            # Get relevant data
            vendor_stats = self._get_all_vendor_statistics(limit=50)
            
            # Prepare context for LLM
            context = {
                'vendor_data': vendor_stats,
                'total_vendors': len(vendor_stats),
                'total_spending': sum(v.get('total_spending', 0) for v in vendor_stats)
            }
            
            # Use LLM to generate recommendations
            if self.use_llm and self.decomposer:
                # Get entities to understand focus areas
                entities = self.decomposer.extract_entities(question)
                
                # Filter data if specific vendors mentioned
                if entities.vendors:
                    filtered_stats = [v for v in vendor_stats 
                                    if any(vendor in v.get('vendor', '') 
                                          for vendor in entities.vendors)]
                    if filtered_stats:
                        vendor_stats = filtered_stats
                
                # Generate business recommendations
                recommendations = self._generate_llm_recommendations(question, vendor_stats)
            else:
                # Fallback to basic recommendations
                recommendations = self._generate_basic_recommendations(vendor_stats)
            
            return {
                "source": "Recommendation Engine",
                "query_type": "recommendation",
                "answer": recommendations,
                "vendors_analyzed": len(vendor_stats),
                "confidence": 85,
                "evidence_report": "Based on comprehensive vendor analysis and business logic"
            }
            
        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
            return self._handle_semantic_search(question)

    def _handle_complex_analytical(self, question: str) -> Dict[str, Any]:
        """
        Handle complex multi-step analytical queries
        """
        if not self.use_llm or not self.decomposer:
            # Fall back to semantic search
            return self._handle_semantic_search(question)
        
        try:
            # Get query decomposition
            query_analysis = decompose_query(question)
            decomposition = query_analysis['decomposition']
            
            # Execute sub-queries
            results = {}
            combined_answer = []
            total_records = 0
            
            for idx, sub_query in enumerate(decomposition['sub_queries']):
                sub_question = sub_query['query']
                
                # Process each sub-query
                if sub_query['type'] == 'sql':
                    sub_result = self.process_query(sub_question)
                else:
                    sub_result = self._handle_semantic_search(sub_question)
                
                results[idx] = sub_result
                
                if sub_result.get('answer'):
                    combined_answer.append(f"**Part {idx+1}:** {sub_result['answer']}")
                
                if sub_result.get('records_analyzed'):
                    total_records += sub_result['records_analyzed']
            
            # Combine results
            final_answer = '\n\n'.join(combined_answer)
            
            # Generate synthesis
            if self.decomposer:
                synthesis = generate_response(question, {'sub_results': results})
                final_answer = synthesis
            
            return {
                'source': 'Complex Analysis',
                'query_type': 'complex_analytical',
                'answer': final_answer,
                'sub_queries_executed': len(decomposition['sub_queries']),
                'records_analyzed': total_records,
                'confidence': 90,
                'decomposition': decomposition
            }
            
        except Exception as e:
            logger.error(f"Complex analysis failed: {e}")
            return self._handle_semantic_search(question)

    def _generate_llm_recommendations(self, question: str, vendor_stats: List[Dict]) -> str:
        """Generate recommendations using LLM"""
        # Prepare structured data for LLM
        vendor_summary = []
        for vendor in vendor_stats[:20]:  # Top 20 for context
            vendor_summary.append({
                'name': vendor.get('vendor'),
                'spending': vendor.get('total_spending', 0),
                'orders': vendor.get('order_count', 0),
                'avg_order': vendor.get('avg_order', 0)
            })
        
        context = {
            'question': question,
            'vendor_data': vendor_summary,
            'analysis_type': 'recommendation'
        }
        
        # Generate response
        response = generate_response(question, context)
        return response

    def _generate_basic_recommendations(self, vendor_stats: List[Dict]) -> str:
        """Generate basic recommendations without LLM"""
        if not vendor_stats:
            return "No vendor data available for recommendations."
        
        # Calculate metrics
        avg_spending = np.mean([v.get('total_spending', 0) for v in vendor_stats])
        total_spending = sum(v.get('total_spending', 0) for v in vendor_stats)
        
        recommendations = ["### Procurement Recommendations\n"]
        
        # Identify top performers
        top_vendors = vendor_stats[:5]
        recommendations.append("**Top Vendors by Spending:**")
        for i, vendor in enumerate(top_vendors, 1):
            pct = (vendor.get('total_spending', 0) / total_spending * 100) if total_spending > 0 else 0
            recommendations.append(f"{i}. {vendor.get('vendor')}: ${vendor.get('total_spending', 0):,.2f} ({pct:.1f}%)")
        
        # Identify optimization opportunities
        low_activity = [v for v in vendor_stats if v.get('order_count', 0) < 5]
        if low_activity:
            recommendations.append(f"\n**Consolidation Opportunities:**")
            recommendations.append(f"{len(low_activity)} vendors have fewer than 5 orders - consider consolidation.")
        
        return '\n'.join(recommendations)

    def _get_all_vendor_statistics(self, limit: int = 100) -> List[Dict]:
        """Get comprehensive statistics for all vendors"""
        if not self.sql_conn:
            return []
        
        try:
            query = f"""
            SELECT 
                {self.VENDOR_COL} as vendor,
                COUNT(*) as order_count,
                SUM(CAST({self.COST_COL} AS FLOAT)) as total_spending,
                AVG(CAST({self.COST_COL} AS FLOAT)) as avg_order,
                MIN(CAST({self.COST_COL} AS FLOAT)) as min_order,
                MAX(CAST({self.COST_COL} AS FLOAT)) as max_order
            FROM procurement
            WHERE {self.COST_COL} IS NOT NULL
            AND {self.VENDOR_COL} IS NOT NULL
            GROUP BY {self.VENDOR_COL}
            ORDER BY total_spending DESC
            LIMIT ?
            """
            
            df = pd.read_sql_query(query, self.sql_conn, params=[limit])
            return df.to_dict('records')
            
        except Exception as e:
            logger.error(f"Failed to get vendor statistics: {e}")
            return []

    # ============================================
    # ORIGINAL HANDLERS (Enhanced with better formatting)
    # ============================================

    def _safe_float(self, value: Any) -> float:
        """Safely convert value to float"""
        if value is None:
            return 0.0
        try:
            if isinstance(value, (int, float)):
                return float(value)
            return float(str(value).replace(',', '').replace('$', ''))
        except:
            return 0.0

    def _safe_int(self, value: Any) -> int:
        """Safely convert value to int"""
        if value is None:
            return 0
        try:
            if isinstance(value, (int, np.integer)):
                return int(value)
            return int(float(str(value).replace(',', '')))
        except:
            return 0

    def _handle_comparison(self, question: str) -> Dict[str, Any]:
        """Enhanced vendor comparison with natural language output"""
        vendors = self._extract_vendor_names(question)
        
        if len(vendors) < 2:
            vendors = self._extract_vendors_with_patterns(question)
        
        if len(vendors) < 2:
            return self._handle_semantic_search(question)
        
        logger.info(f"Comparing vendors: {vendors}")
        
        try:
            results = []
            for vendor in vendors[:10]:
                vendor_data = self._get_vendor_statistics(vendor)
                if vendor_data:
                    results.append(vendor_data)
            
            if not results:
                return {
                    "source": "SQL",
                    "query_type": "comparison",
                    "answer": "No data found for the specified vendors",
                    "confidence": 0
                }
            
            # Format comparison results
            formatted_answer = self._format_comparison_results_enhanced(results)
            
            return {
                "source": "SQL",
                "query_type": "comparison",
                "answer": formatted_answer,
                "vendors": results,
                "records_analyzed": len(results),
                "confidence": 100,
                "evidence_report": "Data Source: SQL Database (Complete Dataset)\nQuery Type: comparison\nConfidence: 100% (deterministic query)"
            }
            
        except Exception as e:
            logger.error(f"Comparison failed: {e}")
            return self._handle_semantic_search(question)

    def _format_comparison_results_enhanced(self, results: List[Dict]) -> str:
        """Enhanced formatting with insights"""
        output = "### Vendor Comparison Analysis\n\n"
        
        # Sort by total spending
        results.sort(key=lambda x: x.get('total_spending', 0), reverse=True)
        
        total_combined = sum(r.get('total_spending', 0) for r in results)
        total_orders = sum(r.get('order_count', 0) for r in results)
        
        # Individual vendor details
        for i, vendor_data in enumerate(results, 1):
            output += f"**{i}. {vendor_data['vendor']}**\n"
            output += f"   * Total Spending: ${vendor_data['total_spending']:,.2f}\n"
            output += f"   * Number of Orders: {vendor_data['order_count']:,}\n"
            output += f"   * Average Order Value: ${vendor_data['avg_order']:,.2f}\n"
            output += f"   * Order Range: ${vendor_data['min_order']:,.2f} - ${vendor_data['max_order']:,.2f}\n\n"
        
        # Summary insights
        output += "---\n### Summary Insights\n\n"
        output += f"**Combined Metrics:**\n"
        output += f"   * Total Combined Spending: ${total_combined:,.2f}\n"
        output += f"   * Total Combined Orders: {total_orders:,}\n"
        output += f"   * Number of Vendors Compared: {len(results)}\n\n"
        
        # Spending distribution
        if total_combined > 0:
            output += "**Spending Distribution:**\n"
            for vendor_data in results:
                percentage = (vendor_data['total_spending'] / total_combined) * 100
                bar = '█' * int(percentage / 5) + '░' * (20 - int(percentage / 5))
                output += f"   * {vendor_data['vendor']}: [{bar}] {percentage:.1f}%\n"
        
        # Winner analysis
        if len(results) > 1:
            output += f"\n**Key Findings:**\n"
            winner = results[0]
            output += f"   * Highest Spending: {winner['vendor']} (${winner['total_spending']:,.2f})\n"
            
            # Find best average order value
            best_avg = max(results, key=lambda x: x.get('avg_order', 0))
            output += f"   * Best Average Order Value: {best_avg['vendor']} (${best_avg['avg_order']:,.2f})\n"
            
            # Find most orders
            most_orders = max(results, key=lambda x: x.get('order_count', 0))
            output += f"   * Most Orders: {most_orders['vendor']} ({most_orders['order_count']:,} orders)\n"
        
        return output

    def _extract_vendors_with_patterns(self, question: str) -> List[str]:
        """Extract vendors using various patterns"""
        vendors = []
        
        patterns = [
            r'(\w+)\s+(?:and|vs\.?|versus|against)\s+(\w+)',
            r'compare\s+(\w+)\s+(?:and|with|to)\s+(\w+)',
            r'(\w+)\s*,\s*(\w+)(?:\s+and\s+(\w+))?'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, question, re.IGNORECASE)
            for match in matches:
                for group in match:
                    if group:
                        potential_vendors = self._find_vendor_in_db(group.upper())
                        vendors.extend(potential_vendors)
        
        return list(set(vendors))[:10]

    def _get_vendor_statistics(self, vendor: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive statistics for a vendor"""
        if not self.sql_conn:
            return None
        
        try:
            vendor_variations = self._get_vendor_variations(vendor)
            
            if not vendor_variations:
                vendor_variations = [vendor]
            
            placeholders = ','.join(['?' for _ in vendor_variations])
            
            query = f"""
            SELECT 
                MIN({self.VENDOR_COL}) as vendor_name,
                COUNT(*) as order_count,
                SUM(CAST({self.COST_COL} AS FLOAT)) as total_spending,
                AVG(CAST({self.COST_COL} AS FLOAT)) as avg_order,
                MIN(CAST({self.COST_COL} AS FLOAT)) as min_order,
                MAX(CAST({self.COST_COL} AS FLOAT)) as max_order
            FROM procurement
            WHERE {self.VENDOR_COL} IN ({placeholders})
            AND {self.COST_COL} IS NOT NULL
            """
            
            df = pd.read_sql_query(query, self.sql_conn, params=vendor_variations)
            
            if not df.empty and df['total_spending'].iloc[0] is not None:
                row = df.iloc[0]
                return {
                    "vendor": row['vendor_name'] or vendor,
                    "total_spending": self._safe_float(row['total_spending']),
                    "order_count": self._safe_int(row['order_count']),
                    "avg_order": self._safe_float(row['avg_order']),
                    "min_order": self._safe_float(row['min_order']),
                    "max_order": self._safe_float(row['max_order'])
                }
        
        except Exception as e:
            logger.error(f"Failed to get vendor statistics for {vendor}: {e}")
        
        return None

    def _get_vendor_variations(self, vendor: str) -> List[str]:
        """Get all variations of a vendor name"""
        variations = [vendor]
        
        # Add from known mappings
        for key, aliases in self.KNOWN_VENDOR_MAPPINGS.items():
            if vendor.upper() in [a.upper() for a in aliases]:
                variations.extend(aliases)
        
        # Add from database mappings
        normalized = self._normalize_vendor_name(vendor)
        if normalized in self.VENDOR_ALIASES:
            variations.extend(self.VENDOR_ALIASES[normalized])
        
        # Search in database
        db_vendors = self._find_vendor_in_db(vendor)
        variations.extend(db_vendors)
        
        return list(set(variations))[:20]

    def _handle_aggregation(self, question: str) -> Dict[str, Any]:
        """Handle aggregation queries with enhanced formatting"""
        question_lower = question.lower()
        
        try:
            if 'total' in question_lower or 'sum' in question_lower:
                query = f"""
                SELECT 
                    SUM(CAST({self.COST_COL} AS FLOAT)) as total_spending,
                    COUNT(*) as record_count
                FROM procurement
                WHERE {self.COST_COL} IS NOT NULL
                """
                
                df = pd.read_sql_query(query, self.sql_conn)
                
                if not df.empty:
                    total = self._safe_float(df['total_spending'].iloc[0])
                    count = self._safe_int(df['record_count'].iloc[0])
                    
                    answer = f"### Total Procurement Analysis\n\n"
                    answer += f"**Total Spending:** ${total:,.2f}\n"
                    answer += f"**Total Records:** {count:,}\n"
                    answer += f"**Average per Record:** ${(total/count):,.2f}" if count > 0 else ""
                    
                    return {
                        "source": "SQL",
                        "query_type": "aggregation",
                        "answer": answer,
                        "records_analyzed": count,
                        "confidence": 100
                    }
            
            elif 'average' in question_lower or 'mean' in question_lower:
                vendors = self._extract_vendor_names(question)
                
                if vendors:
                    vendor_stats = []
                    for vendor in vendors:
                        stats = self._get_vendor_statistics(vendor)
                        if stats:
                            vendor_stats.append(stats)
                    
                    if vendor_stats:
                        avg_of_avgs = np.mean([v['avg_order'] for v in vendor_stats])
                        answer = f"### Average Order Analysis\n\n"
                        answer += f"**Average Order Value:** ${avg_of_avgs:,.2f}\n"
                        answer += f"**Vendors Analyzed:** {len(vendor_stats)}\n\n"
                        
                        for v in vendor_stats:
                            answer += f"   * {v['vendor']}: ${v['avg_order']:,.2f}\n"
                    else:
                        answer = "No data found for the specified vendors."
                else:
                    query = f"""
                    SELECT 
                        AVG(CAST({self.COST_COL} AS FLOAT)) as avg_value,
                        COUNT(*) as record_count
                    FROM procurement
                    WHERE {self.COST_COL} IS NOT NULL
                    """
                    
                    df = pd.read_sql_query(query, self.sql_conn)
                    
                    if not df.empty:
                        avg = self._safe_float(df['avg_value'].iloc[0])
                        count = self._safe_int(df['record_count'].iloc[0])
                        answer = f"### Average Order Analysis\n\n"
                        answer += f"**Average Order Value:** ${avg:,.2f}\n"
                        answer += f"**Based on:** {count:,} orders"
                    else:
                        answer = "Unable to calculate average."
                
                return {
                    "source": "SQL",
                    "query_type": "aggregation",
                    "answer": answer,
                    "records_analyzed": count if 'count' in locals() else 0,
                    "confidence": 100
                }
            
        except Exception as e:
            logger.error(f"Aggregation failed: {e}")
            return self._handle_semantic_search(question)

    def _handle_ranking(self, question: str) -> Dict[str, Any]:
        """Handle ranking queries with enhanced presentation"""
        question_lower = question.lower()
        
        # Determine number to return
        n = 10
        numbers = re.findall(r'\d+', question)
        if numbers:
            n = min(int(numbers[0]), 100)
        
        # Determine sort order
        ascending = any(word in question_lower for word in ['bottom', 'smallest', 'lowest', 'least'])
        
        try:
            query = f"""
            SELECT 
                {self.VENDOR_COL} as vendor,
                COUNT(*) as order_count,
                SUM(CAST({self.COST_COL} AS FLOAT)) as total_spending,
                AVG(CAST({self.COST_COL} AS FLOAT)) as avg_order,
                MIN(CAST({self.COST_COL} AS FLOAT)) as min_order,
                MAX(CAST({self.COST_COL} AS FLOAT)) as max_order
            FROM procurement
            WHERE {self.COST_COL} IS NOT NULL
            AND {self.VENDOR_COL} IS NOT NULL
            GROUP BY {self.VENDOR_COL}
            ORDER BY total_spending {'ASC' if ascending else 'DESC'}
            LIMIT ?
            """
            
            df = pd.read_sql_query(query, self.sql_conn, params=[n])
            
            if not df.empty:
                title = f"### {'Bottom' if ascending else 'Top'} {n} Vendors by Spending\n\n"
                answer = title
                
                # Calculate total for percentage
                total_all = df['total_spending'].sum()
                
                for i, row in enumerate(df.itertuples(), 1):
                    pct = (self._safe_float(row.total_spending) / total_all * 100) if total_all > 0 else 0
                    
                    answer += f"**{i}. {row.vendor}**\n"
                    answer += f"   * Total: ${self._safe_float(row.total_spending):,.2f} ({pct:.1f}%)\n"
                    answer += f"   * Orders: {self._safe_int(row.order_count):,}\n"
                    answer += f"   * Average: ${self._safe_float(row.avg_order):,.2f}\n\n"
                
                # Add summary
                answer += "---\n**Summary:**\n"
                answer += f"   * Combined Total: ${total_all:,.2f}\n"
                answer += f"   * Combined Orders: {df['order_count'].sum():,}\n"
                
                return {
                    "source": "SQL",
                    "query_type": "ranking",
                    "answer": answer,
                    "vendors": df.to_dict('records'),
                    "records_analyzed": len(df),
                    "confidence": 100
                }
        
        except Exception as e:
            logger.error(f"Ranking failed: {e}")
            return self._handle_semantic_search(question)

    def _handle_specific_lookup(self, question: str) -> Dict[str, Any]:
        """Handle specific vendor lookup with comprehensive profile"""
        vendors = self._extract_vendor_names(question)
        
        if not vendors:
            return self._handle_semantic_search(question)
        
        vendor = vendors[0]
        vendor_data = self._get_vendor_statistics(vendor)
        
        if not vendor_data:
            return {
                "source": "SQL",
                "query_type": "specific_lookup",
                "answer": f"No data found for vendor: {vendor}",
                "confidence": 0
            }
        
        # Get additional insights
        commodity_data = self._get_vendor_commodities(vendor)
        
        # Format vendor profile
        answer = f"### Vendor Profile: {vendor_data['vendor']}\n\n"
        answer += "**Financial Overview:**\n"
        answer += f"   * Total Spending: ${vendor_data['total_spending']:,.2f}\n"
        answer += f"   * Number of Orders: {vendor_data['order_count']:,.0f}\n"
        answer += f"   * Average Order Value: ${vendor_data['avg_order']:,.2f}\n"
        answer += f"   * Order Value Range: ${vendor_data['min_order']:,.2f} - ${vendor_data['max_order']:,.2f}\n\n"
        
        if commodity_data:
            answer += "**Top Categories:**\n"
            for comm in commodity_data[:5]:
                answer += f"   * {comm['commodity']}: ${comm['total']:,.2f} ({comm['count']} orders)\n"
        
        return {
            "source": "SQL",
            "query_type": "specific_lookup",
            "answer": answer,
            "vendor_data": vendor_data,
            "commodity_data": commodity_data,
            "records_analyzed": vendor_data['order_count'],
            "confidence": 100
        }

    def _get_vendor_commodities(self, vendor: str) -> List[Dict]:
        """Get commodity breakdown for a vendor"""
        if not self.sql_conn:
            return []
        
        try:
            vendor_variations = self._get_vendor_variations(vendor)
            placeholders = ','.join(['?' for _ in vendor_variations])
            
            query = f"""
            SELECT 
                {self.COMMODITY_COL} as commodity,
                COUNT(*) as count,
                SUM(CAST({self.COST_COL} AS FLOAT)) as total
            FROM procurement
            WHERE {self.VENDOR_COL} IN ({placeholders})
            AND {self.COMMODITY_COL} IS NOT NULL
            AND {self.COST_COL} IS NOT NULL
            GROUP BY {self.COMMODITY_COL}
            ORDER BY total DESC
            LIMIT 10
            """
            
            df = pd.read_sql_query(query, self.sql_conn, params=vendor_variations)
            return df.to_dict('records')
            
        except Exception as e:
            logger.error(f"Failed to get commodity data: {e}")
            return []

    def _handle_statistical(self, question: str) -> Dict[str, Any]:
        """Handle statistical queries with comprehensive analysis"""
        question_lower = question.lower()
        
        # Identify the statistical function needed
        stat_function = None
        for keyword, sql_func in self.STATISTICAL_FUNCTIONS.items():
            if keyword in question_lower:
                stat_function = sql_func
                break
        
        if not stat_function:
            return self._handle_semantic_search(question)
        
        try:
            query = f"""
            SELECT CAST({self.COST_COL} AS FLOAT) as cost
            FROM procurement
            WHERE {self.COST_COL} IS NOT NULL
            AND CAST({self.COST_COL} AS FLOAT) > 0
            """
            
            df = pd.read_sql_query(query, self.sql_conn)
            
            if df.empty:
                return {
                    "source": "SQL",
                    "query_type": "statistical",
                    "answer": "No data available for statistical calculation",
                    "confidence": 0
                }
            
            costs = df['cost'].values
            
            # Calculate comprehensive statistics
            stats = {
                'count': len(costs),
                'mean': np.mean(costs),
                'median': np.median(costs),
                'std': np.std(costs),
                'variance': np.var(costs),
                'min': np.min(costs),
                'max': np.max(costs),
                'q25': np.percentile(costs, 25),
                'q75': np.percentile(costs, 75)
            }
            
            # Format answer based on request
            answer = "### Statistical Analysis\n\n"
            
            if 'median' in question_lower:
                answer += f"**Median Order Value:** ${stats['median']:,.2f}\n\n"
            elif 'variance' in question_lower:
                answer += f"**Variance:** ${stats['variance']:,.2f}\n\n"
            elif 'standard deviation' in question_lower or 'stddev' in question_lower:
                answer += f"**Standard Deviation:** ${stats['std']:,.2f}\n\n"
            elif 'percentile' in question_lower:
                percentile_match = re.search(r'(\d+)(?:th|st|nd|rd)?\s*percentile', question_lower)
                percentile = int(percentile_match.group(1)) if percentile_match else 50
                result = np.percentile(costs, percentile)
                answer += f"**{percentile}th Percentile:** ${result:,.2f}\n\n"
            else:
                answer += f"**Mean:** ${stats['mean']:,.2f}\n\n"
            
            # Add context
            answer += "**Additional Context:**\n"
            answer += f"   * Data Points: {stats['count']:,}\n"
            answer += f"   * Range: ${stats['min']:,.2f} - ${stats['max']:,.2f}\n"
            answer += f"   * Interquartile Range: ${stats['q25']:,.2f} - ${stats['q75']:,.2f}\n"
            
            # Add insights
            if stats['median'] < stats['mean'] * 0.8:
                answer += "\n**Insight:** The median is significantly lower than the mean, "
                answer += "indicating right-skewed distribution with high-value outliers."
            elif stats['median'] > stats['mean'] * 1.2:
                answer += "\n**Insight:** The median is higher than the mean, "
                answer += "suggesting left-skewed distribution with low-value outliers."
            
            return {
                "source": "SQL",
                "query_type": "statistical",
                "answer": answer,
                "statistics": stats,
                "records_analyzed": len(costs),
                "confidence": 100
            }
        
        except Exception as e:
            logger.error(f"Statistical calculation failed: {e}")
            return self._handle_semantic_search(question)

    def _handle_trend_analysis(self, question: str) -> Dict[str, Any]:
        """Handle trend analysis queries"""
        return {
            "source": "SQL",
            "query_type": "trend_analysis",
            "answer": "Trend analysis requires temporal data (dates/times) which may not be available in the current dataset. Consider adding date columns to enable time-based analysis.",
            "confidence": 50
        }

    def _handle_fuzzy_search(self, question: str) -> Dict[str, Any]:
        """Handle fuzzy/pattern matching queries"""
        question_lower = question.lower()
        
        pattern = None
        if 'starting with' in question_lower:
            match = re.search(r'starting with\s+[\'"]?(\w+)', question_lower)
            if match:
                pattern = f"{match.group(1).upper()}%"
        elif 'ending with' in question_lower:
            match = re.search(r'ending with\s+[\'"]?(\w+)', question_lower)
            if match:
                pattern = f"%{match.group(1).upper()}"
        elif 'containing' in question_lower or 'contains' in question_lower:
            match = re.search(r'contain(?:ing|s)\s+[\'"]?(\w+)', question_lower)
            if match:
                pattern = f"%{match.group(1).upper()}%"
        
        if not pattern:
            return self._handle_semantic_search(question)
        
        try:
            query = f"""
            SELECT 
                {self.VENDOR_COL} as vendor,
                COUNT(*) as order_count,
                SUM(CAST({self.COST_COL} AS FLOAT)) as total_spending
            FROM procurement
            WHERE UPPER({self.VENDOR_COL}) LIKE ?
            AND {self.COST_COL} IS NOT NULL
            GROUP BY {self.VENDOR_COL}
            ORDER BY total_spending DESC
            LIMIT 20
            """
            
            df = pd.read_sql_query(query, self.sql_conn, params=[pattern])
            
            if df.empty:
                answer = f"No vendors found matching pattern: {pattern}"
            else:
                answer = f"### Vendors Matching Pattern '{pattern}'\n\n"
                for i, row in enumerate(df.itertuples(), 1):
                    answer += f"**{i}. {row.vendor}**\n"
                    answer += f"   * Total: ${self._safe_float(row.total_spending):,.2f}\n"
                    answer += f"   * Orders: {row.order_count}\n\n"
            
            return {
                "source": "SQL",
                "query_type": "fuzzy_search",
                "answer": answer,
                "records_analyzed": len(df),
                "confidence": 90
            }
        
        except Exception as e:
            logger.error(f"Fuzzy search failed: {e}")
            return self._handle_semantic_search(question)

    def _handle_semantic_search(self, question: str) -> Dict[str, Any]:
        """Enhanced semantic search with LLM support"""
        if self.use_llm and self.decomposer:
            # Try to extract any useful context
            entities = self.decomposer.extract_entities(question)
            
            context_info = []
            if entities.vendors:
                context_info.append(f"Vendors mentioned: {', '.join(entities.vendors)}")
            if entities.metrics:
                context_info.append(f"Metrics of interest: {', '.join(entities.metrics)}")
            
            answer = "This query requires semantic analysis beyond structured SQL queries.\n\n"
            if context_info:
                answer += "**Identified Context:**\n"
                for info in context_info:
                    answer += f"   * {info}\n"
                answer += "\nPlease ensure the RAG system is configured for semantic search capabilities."
            else:
                answer += "Please configure the RAG system for semantic search capabilities."
            
            return {
                "source": "Semantic Search Required",
                "query_type": "semantic_search",
                "answer": answer,
                "entities": entities.__dict__ if entities else {},
                "confidence": 40
            }
        
        return {
            "source": "Semantic Search Required",
            "query_type": "semantic_search",
            "answer": "This query requires semantic search capabilities. Please ensure the RAG system is properly configured.",
            "confidence": 30
        }

    def get_database_stats(self) -> Dict[str, Any]:
        """Get enhanced database statistics"""
        if not self.sql_conn:
            return {"error": "Database not connected"}
        
        try:
            stats = {}
            
            # Total records
            query = "SELECT COUNT(*) as total FROM procurement"
            df = pd.read_sql_query(query, self.sql_conn)
            stats['total_records'] = int(df['total'].iloc[0])
            
            # Unique vendors
            query = f"SELECT COUNT(DISTINCT {self.VENDOR_COL}) as vendors FROM procurement"
            df = pd.read_sql_query(query, self.sql_conn)
            stats['unique_vendors'] = int(df['vendors'].iloc[0])
            
            # Total spending
            query = f"SELECT SUM(CAST({self.COST_COL} AS FLOAT)) as total FROM procurement WHERE {self.COST_COL} IS NOT NULL"
            df = pd.read_sql_query(query, self.sql_conn)
            stats['total_spending'] = float(df['total'].iloc[0]) if df['total'].iloc[0] else 0
            
            # Average order value
            query = f"SELECT AVG(CAST({self.COST_COL} AS FLOAT)) as avg FROM procurement WHERE {self.COST_COL} IS NOT NULL"
            df = pd.read_sql_query(query, self.sql_conn)
            stats['average_order_value'] = float(df['avg'].iloc[0]) if df['avg'].iloc[0] else 0
            
            # Date range (if date column exists)
            try:
                query = "SELECT MIN(date_column) as min_date, MAX(date_column) as max_date FROM procurement"
                df = pd.read_sql_query(query, self.sql_conn)
                stats['date_range'] = {
                    'start': str(df['min_date'].iloc[0]),
                    'end': str(df['max_date'].iloc[0])
                }
            except:
                stats['date_range'] = "Date column not available"
            
            # LLM status
            stats['llm_enabled'] = self.use_llm
            stats['llm_available'] = LLM_AVAILABLE
            
            return stats
        
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {"error": str(e)}
"""
hybrid_rag_architecture.py - Enhanced Hybrid RAG System with Centralized VendorResolver
Integrates LLM for query classification, entity resolution, and response generation
UPDATED: Support for template-based prompts and dual parsing modes
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
from template_utils import extract_from_template_response

# Import shared modules
from constants import (
    DB_PATH, VENDOR_COL, COST_COL, DESC_COL, COMMODITY_COL,
    VENDOR_COL_1, VENDOR_COL_2, VENDOR_SEARCH_BOTH, VENDOR_UNION_QUERY_TEMPLATE,
    VENDOR_SUFFIXES, KNOWN_VENDOR_MAPPINGS, 
    STATISTICAL_KEYWORDS, COMPARISON_KEYWORDS, 
    AGGREGATION_KEYWORDS, RANKING_KEYWORDS,
    FUZZY_THRESHOLD, FUZZY_MAX_MATCHES,
    # NEW: Import dynamic prompt functions
    get_grounded_comparison_prompt, get_grounded_recommendation_prompt,
    get_grounded_statistical_prompt, get_grounded_synthesis_prompt,
    # Import template prompts for direct use if needed
    GROUNDED_COMPARISON_PROMPT_TEMPLATE, GROUNDED_RECOMMENDATION_PROMPT_TEMPLATE,
    GROUNDED_STATISTICAL_PROMPT_TEMPLATE, GROUNDED_SYNTHESIS_PROMPT_TEMPLATE,
    # Import vendor resolution configuration
    VENDOR_FUZZY_THRESHOLDS, VENDOR_RESOLUTION_MAX_RESULTS,
    VENDOR_RESOLUTION_STRATEGIES, VENDOR_NAME_VARIATIONS,
    CACHE_TTL_BY_TYPE, CACHE_KEY_PREFIXES,
    FEATURES
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

# Import cache if available
try:
    from simple_cache import QueryCache
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False
    QueryCache = None
    logger.warning("Cache not available")

# ============================================
# TEMPLATE EXTRACTION UTILITIES
# ============================================



def extract_recommendations_template(response_text: str) -> str:
    """Extract and format recommendation template responses"""
    recommendations = []
    
    # Check for insufficient data
    insufficient_match = re.search(r'<INSUFFICIENT_DATA>(.*?)</INSUFFICIENT_DATA>', 
                                 response_text, re.IGNORECASE | re.DOTALL)
    if insufficient_match:
        return insufficient_match.group(1).strip()
    
    # Extract numbered recommendations
    for i in range(1, 11):  # Support up to 10 recommendations
        rec_pattern = f'<REC{i}>\\s*<ACTION>(.*?)</ACTION>\\s*<JUSTIFICATION>(.*?)</JUSTIFICATION>\\s*(?:<PRIORITY>(.*?)</PRIORITY>)?\\s*</REC{i}>'
        match = re.search(rec_pattern, response_text, re.IGNORECASE | re.DOTALL)
        if match:
            action = match.group(1).strip()
            justification = match.group(2).strip()
            priority = match.group(3).strip() if match.group(3) else "Medium"
            recommendations.append({
                'action': action,
                'justification': justification,
                'priority': priority
            })
    
    if recommendations:
        formatted = "### Strategic Recommendations\n\n"
        for i, rec in enumerate(recommendations, 1):
            formatted += f"**{i}. {rec['action']}** (Priority: {rec['priority']})\n"
            formatted += f"   - Justification: {rec['justification']}\n\n"
        return formatted
    
    # Fallback
    return re.sub(r'<[^>]+>', '', response_text).strip()

def extract_comparison_template(response_text: str) -> str:
    """Extract and format comparison template responses"""
    result = []
    
    # Extract summary
    summary_match = re.search(r'<SUMMARY>(.*?)</SUMMARY>', response_text, re.IGNORECASE | re.DOTALL)
    if summary_match:
        result.append(f"**Summary:** {summary_match.group(1).strip()}\n")
    
    # Extract vendor details
    vendors_data = []
    for i in range(1, 11):
        vendor_pattern = f'<VENDOR{i}>\\s*<NAME>(.*?)</NAME>\\s*<PERFORMANCE>(.*?)</PERFORMANCE>\\s*(?:<STRENGTHS>(.*?)</STRENGTHS>)?\\s*(?:<CONCERNS>(.*?)</CONCERNS>)?\\s*</VENDOR{i}>'
        match = re.search(vendor_pattern, response_text, re.IGNORECASE | re.DOTALL)
        if match:
            vendor_info = {
                'name': match.group(1).strip(),
                'performance': match.group(2).strip(),
                'strengths': match.group(3).strip() if match.group(3) else "",
                'concerns': match.group(4).strip() if match.group(4) else ""
            }
            vendors_data.append(vendor_info)
    
    # Format vendor data
    for vendor in vendors_data:
        result.append(f"### {vendor['name']}")
        result.append(f"**Performance:** {vendor['performance']}")
        if vendor['strengths']:
            result.append(f"**Strengths:** {vendor['strengths']}")
        if vendor['concerns']:
            result.append(f"**Concerns:** {vendor['concerns']}")
        result.append("")
    
    # Extract recommendation
    rec_match = re.search(r'<RECOMMENDATION>(.*?)</RECOMMENDATION>', response_text, re.IGNORECASE | re.DOTALL)
    if rec_match:
        result.append(f"**Recommendation:** {rec_match.group(1).strip()}")
    
    return "\n".join(result) if result else re.sub(r'<[^>]+>', '', response_text).strip()

def extract_statistical_template(response_text: str) -> str:
    """Extract and format statistical template responses"""
    result = []
    
    # Extract summary
    summary_match = re.search(r'<SUMMARY>(.*?)</SUMMARY>', response_text, re.IGNORECASE | re.DOTALL)
    if summary_match:
        result.append(f"**Summary:** {summary_match.group(1).strip()}\n")
    
    # Extract findings
    findings = []
    for i in range(1, 11):
        finding_pattern = f'<FINDING{i}>(.*?)</FINDING{i}>'
        match = re.search(finding_pattern, response_text, re.IGNORECASE | re.DOTALL)
        if match:
            findings.append(f"{i}. {match.group(1).strip()}")
    
    if findings:
        result.append("**Key Findings:**")
        result.extend(findings)
        result.append("")
    
    # Extract business impact
    impact_match = re.search(r'<BUSINESS_IMPACT>(.*?)</BUSINESS_IMPACT>', response_text, re.IGNORECASE | re.DOTALL)
    if impact_match:
        result.append(f"**Business Impact:** {impact_match.group(1).strip()}\n")
    
    # Extract recommendations
    rec_match = re.search(r'<RECOMMENDATIONS>(.*?)</RECOMMENDATIONS>', response_text, re.IGNORECASE | re.DOTALL)
    if rec_match:
        result.append(f"**Recommendations:** {rec_match.group(1).strip()}")
    
    return "\n".join(result) if result else re.sub(r'<[^>]+>', '', response_text).strip()

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
    RECOMMENDATION = "recommendation"
    COMPLEX_ANALYTICAL = "complex_analytical"

# ============================================
# CENTRALIZED VENDOR RESOLVER (Unchanged)
# ============================================

class VendorResolver:
    """
    Centralized vendor resolution with consistent fuzzy matching,
    caching, and multiple resolution strategies.
    """
    
    def __init__(self, db_connection: sqlite3.Connection, 
                 known_mappings: Dict[str, List[str]] = None,
                 cache_enabled: bool = True):
        """
        Initialize the VendorResolver with proper attribute initialization.
        
        Args:
            db_connection: SQLite database connection
            known_mappings: Dictionary of known vendor mappings
            cache_enabled: Whether to enable caching
        """
        self.db_conn = db_connection
        self.known_mappings = known_mappings or KNOWN_VENDOR_MAPPINGS
        self.cache_enabled = cache_enabled and CACHE_AVAILABLE and FEATURES.get('central_vendor_resolver', False)
        
        # Initialize cache if enabled
        if self.cache_enabled and QueryCache:
            self.cache = QueryCache(
                max_size=CACHE_TTL_BY_TYPE.get('vendor_resolution', 2000),
                ttl_seconds=CACHE_TTL_BY_TYPE.get('vendor_resolution', 3600)
            )
        else:
            self.cache = None
        
        # FIXED: Initialize variation_patterns attribute
        try:
            from constants import VENDOR_NAME_VARIATIONS
            self.variation_patterns = [re.compile(pattern, re.IGNORECASE) 
                                     for pattern in VENDOR_NAME_VARIATIONS]
        except ImportError:
            # Fallback patterns if constants not available
            fallback_patterns = [
                r'\s+INC\.?$',
                r'\s+LLC\.?$', 
                r'\s+CORP(?:ORATION)?\.?$',
                r'\s+COMPANY$',
                r'\s+CO\.?$',
                r'\s+LTD\.?$',
                r'\s+LIMITED$',
                r'\s+TECHNOLOGIES$',
                r'\s+SYSTEMS$',
                r'\s+SOLUTIONS$',
                r'\s+SERVICES$',
                r'\s+INTERNATIONAL$'
            ]
            self.variation_patterns = [re.compile(pattern, re.IGNORECASE) 
                                     for pattern in fallback_patterns]
        
        # Build vendor lookup tables
        self._build_vendor_tables()
        
        logger.info(f"VendorResolver initialized (cache: {self.cache_enabled}, patterns: {len(self.variation_patterns)})")
    def _build_vendor_tables(self):
        """Build lookup tables for efficient vendor resolution"""
        self.normalized_to_original = {}
        self.vendor_set = set()
        
        try:
            # Get all unique vendors from database
            # Use UNION to get vendors from both columns
            query = f"""
            SELECT DISTINCT vendor_name FROM (
                SELECT {VENDOR_COL_1} as vendor_name FROM procurement WHERE {VENDOR_COL_1} IS NOT NULL
                UNION
                SELECT {VENDOR_COL_2} as vendor_name FROM procurement WHERE {VENDOR_COL_2} IS NOT NULL
            )
            """
            df = pd.read_sql_query(query, self.db_conn)
            
            for vendor in df['vendor_name']:
                if vendor:
                    # Store original
                    self.vendor_set.add(vendor)
                    
                    # Store normalized version
                    normalized = self._normalize_vendor_name(vendor)
                    if normalized not in self.normalized_to_original:
                        self.normalized_to_original[normalized] = []
                    self.normalized_to_original[normalized].append(vendor)
                    
                    # Store base name
                    base = self._extract_base_name(vendor)
                    if base not in self.normalized_to_original:
                        self.normalized_to_original[base] = []
                    self.normalized_to_original[base].append(vendor)
            
            logger.info(f"Built vendor tables: {len(self.vendor_set)} unique vendors")
            
        except Exception as e:
            logger.error(f"Failed to build vendor tables: {e}")
    
    def resolve(self, vendor_input: str, max_results: Optional[int] = None) -> List[str]:
        """
        Main resolution method - tries multiple strategies to find matching vendors.
        
        Args:
            vendor_input: The vendor name to resolve
            max_results: Maximum number of results to return
        
        Returns:
            List of matching vendor names from the database
        """
        if not vendor_input:
            return []
        
        max_results = max_results or VENDOR_RESOLUTION_MAX_RESULTS
        
        # Check cache first
        if self.cache_enabled and self.cache:
            cache_key = f"{CACHE_KEY_PREFIXES['vendor']}{vendor_input.lower()}"
            cached_result = self.cache.get(cache_key)
            if cached_result:
                logger.debug(f"Vendor cache hit for: {vendor_input}")
                return cached_result[:max_results]
        
        results = []
        
        # Try each resolution strategy in order
        for strategy in VENDOR_RESOLUTION_STRATEGIES:
            if strategy == 'exact_match':
                results = self._exact_match(vendor_input)
            elif strategy == 'known_mappings':
                results = self._known_mappings_match(vendor_input)
            elif strategy == 'normalized_match':
                results = self._normalized_match(vendor_input)
            elif strategy == 'fuzzy_match':
                results = self._fuzzy_match(vendor_input)
            elif strategy == 'partial_match':
                results = self._partial_match(vendor_input)
            
            if results:
                logger.debug(f"Vendor '{vendor_input}' resolved using {strategy}: {results[:3]}")
                break
        
        # Deduplicate and limit results
        seen = set()
        unique_results = []
        for vendor in results:
            if vendor not in seen:
                seen.add(vendor)
                unique_results.append(vendor)
                if len(unique_results) >= max_results:
                    break
        
        # Cache the result
        if self.cache_enabled and self.cache and unique_results:
            cache_key = f"{CACHE_KEY_PREFIXES['vendor']}{vendor_input.lower()}"
            self.cache.set(cache_key, unique_results)
        
        return unique_results
    
    def get_canonical_name(self, vendor_input: str) -> Optional[str]:
        """
        Get the single best matching vendor name.
        
        Args:
            vendor_input: The vendor name to resolve
        
        Returns:
            The best matching vendor name or None
        """
        results = self.resolve(vendor_input, max_results=1)
        return results[0] if results else None
    
    def _normalize_vendor_name(self, vendor: str) -> str:
        """Normalize vendor name for matching with proper variation pattern handling"""
        if not vendor:
            return ""
        
        normalized = vendor.upper().strip()
        
        # Remove punctuation except essential ones
        normalized = re.sub(r'[^\w\s\-\&]', ' ', normalized)
        
        # Apply variation patterns to remove common suffixes
        if hasattr(self, 'variation_patterns') and self.variation_patterns:
            for pattern in self.variation_patterns:
                normalized = pattern.sub('', normalized)
        else:
            # Fallback: remove common suffixes manually
            for suffix in ['INC', 'LLC', 'CORP', 'CORPORATION', 'COMPANY', 'CO', 'LTD']:
                normalized = re.sub(r'\b' + suffix + r'\b', '', normalized, flags=re.IGNORECASE)
        
        # Remove extra spaces
        normalized = ' '.join(normalized.split()).strip()
        
        return normalized

    def _extract_base_name(self, vendor: str) -> str:
        """Extract the base vendor name (first significant word)"""
        if not vendor:
            return ""
        
        normalized = self._normalize_vendor_name(vendor)
        words = normalized.split()
        
        # Skip common prefixes
        skip_words = ['THE', 'A', 'AN']
        
        for word in words:
            if word not in skip_words and len(word) > 2:
                return word
        
        return normalized


    def _exact_match(self, vendor_input: str) -> List[str]:
        """Exact match strategy"""
        results = []
        vendor_upper = vendor_input.upper()
        
        # Check exact match (case insensitive)
        for vendor in self.vendor_set:
            if vendor.upper() == vendor_upper:
                results.append(vendor)
        
        return results
    
    def _known_mappings_match(self, vendor_input: str) -> List[str]:
        """Check against known vendor mappings"""
        vendor_upper = vendor_input.upper()
        results = []
        
        # Check if input matches any known mapping
        for canonical, aliases in self.known_mappings.items():
            if vendor_upper in [a.upper() for a in aliases]:
                # Find all database vendors matching this canonical name
                for vendor in self.vendor_set:
                    if any(alias.upper() in vendor.upper() for alias in aliases):
                        results.append(vendor)
                        break
        
        return results
    
    def _normalized_match(self, vendor_input: str) -> List[str]:
        """Match using normalized vendor name"""
        normalized = self._normalize_vendor_name(vendor_input)
        
        # Check normalized lookup table
        if normalized in self.normalized_to_original:
            return self.normalized_to_original[normalized].copy()
        
        # Check base name
        base = self._extract_base_name(vendor_input)
        if base in self.normalized_to_original:
            return self.normalized_to_original[base].copy()
        
        return []
    
    def _fuzzy_match(self, vendor_input: str) -> List[str]:
        """Fuzzy matching with configurable thresholds"""
        results = []
        vendor_upper = vendor_input.upper()
        normalized_input = self._normalize_vendor_name(vendor_input)
        
        for vendor in self.vendor_set:
            vendor_upper_db = vendor.upper()
            vendor_normalized = self._normalize_vendor_name(vendor)
            
            # Calculate similarity scores
            exact_similarity = SequenceMatcher(None, vendor_upper, vendor_upper_db).ratio()
            normalized_similarity = SequenceMatcher(None, normalized_input, vendor_normalized).ratio()
            
            # Use the higher similarity score
            similarity = max(exact_similarity, normalized_similarity)
            
            if similarity >= VENDOR_FUZZY_THRESHOLDS.get('medium_confidence', 0.8):
                results.append(vendor)
        
        return results[:VENDOR_RESOLUTION_MAX_RESULTS]
    
    def _partial_match(self, vendor_input: str) -> List[str]:
        """Partial matching - last resort"""
        results = []
        vendor_upper = vendor_input.upper()
        
        # Check if vendor input is contained in any database vendor name
        for vendor in self.vendor_set:
            if vendor_upper in vendor.upper() or vendor.upper() in vendor_upper:
                results.append(vendor)
        
        return results[:VENDOR_RESOLUTION_MAX_RESULTS]
    
    def get_similar_vendors(self, vendor_input: str, threshold: float = 0.6) -> List[Tuple[str, float]]:
        """
        Get similar vendors with similarity scores.
        
        Args:
            vendor_input: The vendor name to find similar matches for
            threshold: Minimum similarity threshold
        
        Returns:
            List of (vendor_name, similarity_score) tuples
        """
        similar = []
        normalized_input = self._normalize_vendor_name(vendor_input)
        
        for vendor in self.vendor_set:
            vendor_normalized = self._normalize_vendor_name(vendor)
            similarity = SequenceMatcher(None, normalized_input, vendor_normalized).ratio()
            
            if similarity >= threshold:
                similar.append((vendor, similarity))
        
        # Sort by similarity
        similar.sort(key=lambda x: x[1], reverse=True)
        
        return similar[:VENDOR_RESOLUTION_MAX_RESULTS]



# ============================================
# PHASE 2: VENDORRESOLVER INTEGRATION
# ============================================

try:
    from vendor_resolver import get_vendor_resolver
    VENDOR_RESOLVER_AVAILABLE = True
    logger.info("VendorResolver available for hybrid RAG processing")
except ImportError:
    VENDOR_RESOLVER_AVAILABLE = False
    logger.warning("VendorResolver not available for hybrid RAG processing")

class HybridProcurementRAG:
    """Enhanced Hybrid system with LLM-powered intelligence and VendorResolver"""
    
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
        
        # Use column mappings from constants (updated for dual-vendor support)
        self.VENDOR_COL = VENDOR_COL  # Backward compatibility
        self.VENDOR_COL_1 = VENDOR_COL_1
        self.VENDOR_COL_2 = VENDOR_COL_2
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
        
        # Initialize Centralized VendorResolver
        if FEATURES.get('central_vendor_resolver', False):
            self.vendor_resolver = VendorResolver(
                db_connection=self.sql_conn,
                known_mappings=KNOWN_VENDOR_MAPPINGS,
                cache_enabled=FEATURES.get('granular_caching', False)
            )
            logger.info("Using centralized VendorResolver")
        else:
            self.vendor_resolver = None
            # Fall back to old vendor resolution logic
            self.VENDOR_ALIASES, self.VENDOR_MAPPING, self.VENDOR_KEYWORDS = self._generate_comprehensive_vendor_mappings()
            logger.info("Using legacy vendor resolution")
        
        # Cache for vendor lookups and LLM results
        self._vendor_cache = {} if not self.vendor_resolver else None
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
        """Generate comprehensive vendor mappings from database (LEGACY - kept for compatibility)"""
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
                    base_name = self._extract_base_name(vendor)
                    
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
        """Improved vendor name normalization (LEGACY - kept for compatibility)"""
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
        """Extract base vendor name (LEGACY - kept for compatibility)"""
        normalized = self._normalize_vendor_name(vendor)
        words = normalized.split()
        
        # Skip common prefixes
        skip_words = ['THE', 'A', 'AN']
        
        for word in words:
            if word not in skip_words and len(word) > 2:
                return word
        
        return normalized

        def _resolve_vendor_with_fallback(self, vendor_input: str) -> List[str]:
        """
        Resolve vendor using VendorResolver with database fallback
        """
        resolved_vendors = []
        
        # Try VendorResolver first
        if VENDOR_RESOLVER_AVAILABLE:
            try:
                resolver = get_vendor_resolver()
                resolved = resolver.get_canonical_name(vendor_input)
                
                if resolved and resolved != "UNKNOWN":
                    # Get all variations
                    variations = resolver.get_all_variations(resolved)
                    resolved_vendors.extend(variations)
                    logger.info(f"VendorResolver found {len(variations)} variations for '{vendor_input}'")
                
            except Exception as e:
                logger.warning(f"VendorResolver failed for '{vendor_input}': {e}")
        
        # Fallback to original database logic
        if not resolved_vendors:
            resolved_vendors = self._find_vendor_in_db(vendor_input)
        
        # Final fallback to input
        if not resolved_vendors:
            resolved_vendors = [vendor_input]
        
        return resolved_vendors[:5]  # Limit to top 5


    def _classify_query(self, question: str) -> QueryType:
        """
        Enhanced query classification with LLM support
        Falls back to rule-based classification if LLM unavailable
        """
        # LLM-POWERED CLASSIFICATION
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
        
        # FALLBACK: RULE-BASED CLASSIFICATION
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
        Enhanced vendor extraction using centralized VendorResolver
        """
        # Use VendorResolver if available
        if self.vendor_resolver and FEATURES.get('central_vendor_resolver', False):
            vendors = []
            
            # First try LLM extraction if available
            if self.use_llm and self.decomposer:
                try:
                    entities = self.decomposer.extract_entities(question)
                    
                    # Resolve each extracted vendor
                    for vendor in entities.vendors:
                        resolved = self.vendor_resolver.resolve(vendor)
                        vendors.extend(resolved)
                    
                    # Handle ambiguous references
                    if entities.ambiguous_references:
                        for ref, suggested in entities.ambiguous_references.items():
                            resolved = self.vendor_resolver.resolve(ref)
                            if not resolved and suggested:
                                resolved = self.vendor_resolver.resolve(suggested)
                            vendors.extend(resolved)
                    
                    if vendors:
                        logger.info(f"VendorResolver + LLM extracted: {vendors[:5]}")
                        return list(set(vendors))[:10]
                        
                except Exception as e:
                    logger.warning(f"LLM entity extraction failed: {e}")
            
            # Fall back to pattern-based extraction with VendorResolver
            question_upper = question.upper()
            words = question.split()
            
            for word in words:
                if len(word) > 3:  # Skip short words
                    resolved = self.vendor_resolver.resolve(word)
                    vendors.extend(resolved)
            
            # Also check known mappings
            for vendor_key, aliases in KNOWN_VENDOR_MAPPINGS.items():
                for alias in aliases:
                    if alias in question_upper:
                        resolved = self.vendor_resolver.resolve(vendor_key)
                        vendors.extend(resolved)
            
            return list(set(vendors))[:10]
        
        # FALLBACK: Original extraction logic
        return self._extract_vendor_names_original(question)

    def _extract_vendor_names_original(self, question: str) -> List[str]:
        """Original vendor extraction logic (LEGACY)"""
        vendors = []
        question_upper = question.upper()
        
        # First check known vendor mappings
        for vendor_key, aliases in self.KNOWN_VENDOR_MAPPINGS.items():
            for alias in aliases:
                if alias in question_upper:
                    actual_vendors = self._find_vendor_in_db(vendor_key)
                    vendors.extend(actual_vendors)
        
        # Check against database vendors
        if hasattr(self, 'VENDOR_MAPPING'):
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
        # Use VendorResolver if available
        if self.vendor_resolver and FEATURES.get('central_vendor_resolver', False):
            return self.vendor_resolver.resolve(vendor_key, max_results=5)
        
        # LEGACY implementation
        if self._vendor_cache is not None and vendor_key in self._vendor_cache:
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
            if self._vendor_cache is not None:
                self._vendor_cache[vendor_key] = vendors[:5]
            
        except Exception as e:
            logger.error(f"Error finding vendor {vendor_key}: {e}")
        
        return vendors

    def _find_fuzzy_vendor_matches(self, search_term: str) -> List[str]:
        """Find vendors using fuzzy matching"""
        # Use VendorResolver if available
        if self.vendor_resolver and FEATURES.get('central_vendor_resolver', False):
            return self.vendor_resolver.resolve(search_term, max_results=3)
        
        # LEGACY implementation
        matches = []
        search_upper = search_term.upper()
        
        if hasattr(self, 'VENDOR_MAPPING'):
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
            
            # Handle LLM-SPECIFIC QUERY TYPES
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
            
            # Enhance results with natural language if LLM available
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
        UPDATED: Now handles template responses
        """
        if not self.use_llm or not self.decomposer:
            return result
        
        try:
            # Use grounded prompts if enabled
            if FEATURES.get('grounded_prompts', False):
                # Generate natural language response with grounded template
                original_answer = result.get('answer', '')
                enhanced_answer = generate_response(question, result)
                
                # Extract template content if template parsing is enabled
                if FEATURES.get('template_parsing', False):
                    enhanced_answer = extract_from_template_response(enhanced_answer)
            else:
                # Use original enhancement
                original_answer = result.get('answer', '')
                enhanced_answer = generate_response(question, result)
            
            # Keep both versions
            result['raw_answer'] = original_answer
            result['answer'] = enhanced_answer
            result['llm_enhanced'] = True
            result['template_parsing'] = FEATURES.get('template_parsing', False)
            
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

    # NEW HANDLER METHODS

    def _handle_recommendation(self, question: str) -> Dict[str, Any]:
        """
        Handle recommendation queries using LLM and data analysis
        UPDATED: Now uses dynamic recommendation prompt with template support
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
                    # Use VendorResolver to resolve vendor names
                    resolved_vendors = []
                    for vendor in entities.vendors:
                        if self.vendor_resolver:
                            resolved = self.vendor_resolver.resolve(vendor)
                            resolved_vendors.extend(resolved)
                        else:
                            resolved_vendors.append(vendor)
                    
                    filtered_stats = [v for v in vendor_stats 
                                    if any(vendor in v.get('vendor', '') 
                                          for vendor in resolved_vendors)]
                    if filtered_stats:
                        vendor_stats = filtered_stats
                
                # Generate business recommendations with template support
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
                "evidence_report": "Based on comprehensive vendor analysis and business logic",
                "template_parsing": FEATURES.get('template_parsing', False)
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
                sub_question = sub_query['query'] if isinstance(sub_query, dict) else sub_query
                
                # Process each sub-query
                if isinstance(sub_query, dict) and sub_query.get('type') == 'sql':
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
                
                # Extract template content if needed
                if FEATURES.get('template_parsing', False):
                    synthesis = extract_from_template_response(synthesis)
                
                final_answer = synthesis
            
            return {
                'source': 'Complex Analysis',
                'query_type': 'complex_analytical',
                'answer': final_answer,
                'sub_queries_executed': len(decomposition['sub_queries']),
                'records_analyzed': total_records,
                'confidence': 90,
                'decomposition': decomposition,
                'template_parsing': FEATURES.get('template_parsing', False)
            }
            
        except Exception as e:
            logger.error(f"Complex analysis failed: {e}")
            return self._handle_semantic_search(question)

    def _generate_llm_recommendations(self, question: str, vendor_stats: List[Dict]) -> str:
        """
        Generate recommendations using LLM
        UPDATED: Now uses dynamic prompt with template support
        """
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
        
        # Use dynamic recommendation prompt
        prompt = get_grounded_recommendation_prompt().format(
            context=json.dumps(context, indent=2),
            focus=question
        )
        
        # Generate response
        response = generate_response(question, context)
        
        # Extract template content if template parsing is enabled
        if FEATURES.get('template_parsing', False):
            response = extract_from_template_response(response)
        
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
                vendor_name as vendor,
                COUNT(*) as order_count,
                SUM(CAST({self.COST_COL} AS FLOAT)) as total_spending,
                AVG(CAST({self.COST_COL} AS FLOAT)) as avg_order,
                MIN(CAST({self.COST_COL} AS FLOAT)) as min_order,
                MAX(CAST({self.COST_COL} AS FLOAT)) as max_order
            FROM (
                SELECT {self.VENDOR_COL_1} as vendor_name, {self.COST_COL} FROM procurement WHERE {self.VENDOR_COL_1} IS NOT NULL
                UNION ALL
                SELECT {self.VENDOR_COL_2} as vendor_name, {self.COST_COL} FROM procurement WHERE {self.VENDOR_COL_2} IS NOT NULL
            )
            WHERE {self.COST_COL} IS NOT NULL
            GROUP BY vendor_name
            ORDER BY total_spending DESC
            LIMIT ?
            """
            
            df = pd.read_sql_query(query, self.sql_conn, params=[limit])
            return df.to_dict('records')
            
        except Exception as e:
            logger.error(f"Failed to get vendor statistics: {e}")
            return []

    def _handle_statistical(self, question: str) -> Dict[str, Any]:
        """
        Handles statistical queries by fetching data and using helper functions.
        Example: "What is the median spending for Dell?"
        """
        logger.info(f"Handling statistical query: {question}")
        question_lower = question.lower()
        vendors = self._extract_vendor_names(question)

        # Determine the statistical function
        metric = "all"
        # STATISTICAL_KEYWORDS is from constants.py
        for keyword in STATISTICAL_KEYWORDS:
            if keyword in question_lower:
                metric = keyword
                break

        # 'average' is a common synonym for 'mean'
        if 'average' in question_lower:
            metric = 'mean'

        # Build query
        query = f"SELECT CAST({self.COST_COL} AS FLOAT) as value FROM procurement"
        params = []
        where_clauses = [f"{self.COST_COL} IS NOT NULL"]

        if vendors:
            # Use VendorResolver if available to get all possible names
            if self.vendor_resolver:
                resolved_vendors = []
                for v_name in vendors:
                    resolved_vendors.extend(self.vendor_resolver.resolve(v_name))
                vendors = list(set(resolved_vendors))

            if vendors:
                placeholders = ','.join(['?' for _ in vendors])
                where_clauses.append(f"{self.VENDOR_COL} IN ({placeholders})")
                params.extend(vendors)

        if where_clauses:
            query += " WHERE " + " AND ".join(where_clauses)

        try:
            df = safe_execute_query(query, params)
            if df.empty:
                return {"answer": "No data found for the specified criteria."}

            values = df['value'].dropna().values

            # MIN_DATA_REQUIREMENTS and INSUFFICIENT_DATA_MESSAGES are from constants.py
            if len(values) < MIN_DATA_REQUIREMENTS.get('statistical', 10):
                 return {
                    "warning": INSUFFICIENT_DATA_MESSAGES['insufficient_samples'].format(
                        required=MIN_DATA_REQUIREMENTS.get('statistical', 10),
                        found=len(values)
                    ),
                    "answer": "Not enough data to perform a reliable statistical calculation."
                }

            # This function is defined in app_helpers.py, but we copy it here to avoid circular import
            stats_result = self._calculate_statistical_metrics(values, metric)

            # Create a user-friendly answer
            answer = f"Statistical analysis for vendors: {', '.join(vendors) if vendors else 'All Vendors'}\n"
            if "value" in stats_result:
                answer += f"The {metric} is: ${stats_result['value']:,.2f}"
            else:
                for key, val in stats_result.items():
                    if key not in ['metric', 'records_analyzed']:
                        answer += f"- {key.capitalize()}: ${val:,.2f}\n"

            return {
                "source": "SQL Query + Python Stats",
                "query_type": "statistical",
                "answer": answer,
                "statistics": stats_result,
                "vendors_analyzed": vendors,
                "records_analyzed": len(values)
            }

        except Exception as e:
            logger.error(f"Statistical query failed: {e}")
            return {"error": str(e), "query": query}


    def _handle_specific_lookup(self, question: str) -> Dict[str, Any]:
        """
        Handle specific vendor lookup queries
        Example: "Tell me about Dell spending"
        """
        logger.info(f"Handling specific lookup query: {question}")
        vendors = self._extract_vendor_names(question)
        
        if not vendors:
            return {
                "source": "Specific Lookup",
                "query_type": "specific_lookup", 
                "answer": "No vendor specified in query",
                "confidence": 0
            }
        
        vendor_name = vendors[0]  # Use first vendor found
        
        try:
            # Use app_helpers function to get comprehensive data
            from app_helpers import get_vendor_comprehensive_data
            vendor_data = get_vendor_comprehensive_data(vendor_name)
            
            if vendor_data:
                answer = f"""Vendor Analysis for {vendor_data.get('vendor', vendor_name)}:
- Total Orders: {vendor_data.get('order_count', 0):,}
- Total Spending: ${vendor_data.get('total_spending', 0):,.2f}
- Average Order Value: ${vendor_data.get('avg_order', 0):,.2f}
- Order Range: ${vendor_data.get('min_order', 0):,.2f} to ${vendor_data.get('max_order', 0):,.2f}"""
                
                return {
                    "source": "Vendor Lookup",
                    "query_type": "specific_lookup",
                    "answer": answer,
                    "vendor_data": vendor_data,
                    "confidence": 95,
                    "records_analyzed": vendor_data.get('order_count', 0)
                }
            else:
                return {
                    "source": "Vendor Lookup", 
                    "query_type": "specific_lookup",
                    "answer": f"No data found for vendor '{vendor_name}'. The vendor may not exist in our database or may be listed under a different name.",
                    "confidence": 0
                }
                
        except Exception as e:
            logger.error(f"Specific lookup failed: {e}")
            return {
                "source": "Vendor Lookup",
                "query_type": "specific_lookup", 
                "answer": f"Error retrieving data for '{vendor_name}': {str(e)}",
                "confidence": 0
            }

    def _handle_comparison(self, question: str) -> Dict[str, Any]:
        """
        Handle vendor comparison queries
        Example: "Compare Dell and Microsoft spending"
        """
        logger.info(f"Handling comparison query: {question}")
        vendors = self._extract_vendor_names(question)
        
        if len(vendors) < 2:
            return {
                "source": "Vendor Comparison",
                "query_type": "comparison",
                "answer": "Please specify at least two vendors to compare",
                "confidence": 0
            }
        
        try:
            # Get data for each vendor
            from app_helpers import get_vendor_comprehensive_data
            vendor_data = []
            
            for vendor in vendors[:5]:  # Limit to 5 vendors max
                data = get_vendor_comprehensive_data(vendor)
                if data:
                    vendor_data.append(data)
            
            if len(vendor_data) < 2:
                return {
                    "source": "Vendor Comparison",
                    "query_type": "comparison", 
                    "answer": f"Could not find sufficient data to compare the requested vendors. Found data for {len(vendor_data)} vendors.",
                    "confidence": 20
                }
            
            # Build comparison response
            answer = "Vendor Spending Comparison:\n\n"
            
            # Sort by spending for ranking
            vendor_data.sort(key=lambda x: x.get('total_spending', 0), reverse=True)
            
            for i, data in enumerate(vendor_data, 1):
                answer += f"{i}. **{data.get('vendor', 'Unknown')}**\n"
                answer += f"   - Total Spending: ${data.get('total_spending', 0):,.2f}\n"
                answer += f"   - Orders: {data.get('order_count', 0):,}\n"
                answer += f"   - Avg Order: ${data.get('avg_order', 0):,.2f}\n\n"
            
            # Add insights
            top_vendor = vendor_data[0]
            total_spending = sum(v.get('total_spending', 0) for v in vendor_data)
            top_percentage = (top_vendor.get('total_spending', 0) / total_spending * 100) if total_spending > 0 else 0
            
            answer += f"**Key Insights:**\n"
            answer += f"- {top_vendor.get('vendor')} leads with {top_percentage:.1f}% of combined spending\n"
            answer += f"- Total combined spending: ${total_spending:,.2f}"
            
            return {
                "source": "Vendor Comparison",
                "query_type": "comparison",
                "answer": answer,
                "vendor_data": vendor_data,
                "confidence": 90,
                "vendors_compared": len(vendor_data),
                "records_analyzed": sum(v.get('order_count', 0) for v in vendor_data)
            }
            
        except Exception as e:
            logger.error(f"Comparison failed: {e}")
            return {
                "source": "Vendor Comparison",
                "query_type": "comparison",
                "answer": f"Error comparing vendors: {str(e)}",
                "confidence": 0
            }

    def _handle_aggregation(self, question: str) -> Dict[str, Any]:
        """
        Handle aggregation queries 
        Example: "What is total spending?"
        """
        logger.info(f"Handling aggregation query: {question}")
        
        try:
            # Use database manager for total spending
            from database_utils import db_manager
            stats = db_manager.get_stats()
            
            total_spending = stats.get('total_spending', 0)
            total_orders = stats.get('total_records', 0)
            unique_vendors = stats.get('unique_vendors', 0)
            avg_order = stats.get('average_order', 0)
            
            answer = f"""Procurement Summary:
- Total Spending: ${total_spending:,.2f}
- Total Orders: {total_orders:,}
- Unique Vendors: {unique_vendors:,}
- Average Order Value: ${avg_order:,.2f}"""
            
            return {
                "source": "Database Aggregation",
                "query_type": "aggregation", 
                "answer": answer,
                "statistics": stats,
                "confidence": 100,
                "records_analyzed": total_orders
            }
            
        except Exception as e:
            logger.error(f"Aggregation failed: {e}")
            return {
                "source": "Database Aggregation",
                "query_type": "aggregation",
                "answer": f"Error calculating totals: {str(e)}",
                "confidence": 0
            }

    def _handle_ranking(self, question: str) -> Dict[str, Any]:
        """
        Handle ranking queries
        Example: "Who are the top 5 vendors?"
        """
        logger.info(f"Handling ranking query: {question}")
        
        try:
            # Get top vendors using dual-column helper
            from app_helpers import get_vendor_stats_both_columns
            vendor_stats = get_vendor_stats_both_columns()
            
            if not vendor_stats:
                return {
                    "source": "Vendor Ranking",
                    "query_type": "ranking",
                    "answer": "No vendor data available for ranking",
                    "confidence": 0
                }
            
            # Take top 10
            top_vendors = vendor_stats[:10]
            
            answer = "Top Vendors by Spending:\n\n"
            for i, vendor in enumerate(top_vendors, 1):
                answer += f"{i}. **{vendor.get('vendor_name', 'Unknown')}**\n"
                answer += f"   - Spending: ${vendor.get('total_spending', 0):,.2f}\n"
                answer += f"   - Orders: {vendor.get('order_count', 0):,}\n\n"
            
            total_top_spending = sum(v.get('total_spending', 0) for v in top_vendors)
            answer += f"**Combined top {len(top_vendors)} spending: ${total_top_spending:,.2f}**"
            
            return {
                "source": "Vendor Ranking",
                "query_type": "ranking",
                "answer": answer,
                "vendor_data": top_vendors,
                "confidence": 95,
                "vendors_ranked": len(top_vendors)
            }
            
        except Exception as e:
            logger.error(f"Ranking failed: {e}")
            return {
                "source": "Vendor Ranking", 
                "query_type": "ranking",
                "answer": f"Error generating vendor ranking: {str(e)}",
                "confidence": 0
            }

    def _handle_fuzzy_search(self, question: str) -> Dict[str, Any]:
        """
        Handle fuzzy search queries
        Example: "Find vendors like Dell"
        """
        logger.info(f"Handling fuzzy search query: {question}")
        
        try:
            vendors = self._extract_vendor_names(question)
            if not vendors:
                return {
                    "source": "Fuzzy Search",
                    "query_type": "fuzzy_search",
                    "answer": "No vendor pattern specified for fuzzy search",
                    "confidence": 0
                }
            
            search_term = vendors[0]
            
            # Use VendorResolver for fuzzy matching if available
            if self.vendor_resolver:
                matches = self.vendor_resolver.resolve(search_term, max_results=10)
            else:
                # Fallback to basic search
                from app_helpers import search_vendors_both_columns
                matches = search_vendors_both_columns(search_term, limit=10)
            
            if matches:
                answer = f"Vendors similar to '{search_term}':\n\n"
                for i, match in enumerate(matches, 1):
                    answer += f"{i}. {match}\n"
                
                return {
                    "source": "Fuzzy Search",
                    "query_type": "fuzzy_search", 
                    "answer": answer,
                    "matches": matches,
                    "confidence": 85,
                    "search_term": search_term
                }
            else:
                return {
                    "source": "Fuzzy Search",
                    "query_type": "fuzzy_search",
                    "answer": f"No vendors found similar to '{search_term}'",
                    "confidence": 50
                }
                
        except Exception as e:
            logger.error(f"Fuzzy search failed: {e}")
            return {
                "source": "Fuzzy Search",
                "query_type": "fuzzy_search", 
                "answer": f"Error in fuzzy search: {str(e)}",
                "confidence": 0
            }

    def _handle_semantic_search(self, question: str) -> Dict[str, Any]:
        """
        Handles semantic search queries using database full-text search as a fallback.
        """
        logger.info(f"Handling semantic search (FTS fallback) for: {question}")
        try:
            results_df = db_manager.search_full_text(question, limit=10)

            if results_df.empty:
                return {
                    "answer": "I could not find any specific data related to your query using full-text search.",
                    "source": "FTS Fallback"
                }

            # Format the results
            records = results_df.to_dict('records')
            answer = "Found some potentially relevant records:\n"
            for record in records:
                answer += f"- Vendor: {record.get(self.VENDOR_COL)}, Cost: {record.get(self.COST_COL)}, Desc: {str(record.get(self.DESC_COL))[:50]}...\n"

            return {
                "source": "FTS Fallback",
                "query_type": "semantic_search",
                "answer": answer,
                "data": records
            }

        except Exception as e:
            logger.error(f"Semantic search (FTS) failed: {e}")
            return {"error": str(e), "source": "FTS Fallback"}

    def _calculate_statistical_metrics(self, values: np.ndarray, metric: str = "all") -> Dict:
        """
        Calculate statistical metrics.
        Copied from app_helpers.py to avoid circular imports.
        """
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

# ============================================
# MODULE-LEVEL FUNCTIONS
# ============================================

# Singleton instance for VendorResolver
_vendor_resolver = None

def get_vendor_resolver() -> VendorResolver:
    """Get singleton VendorResolver instance"""
    global _vendor_resolver
    
    if _vendor_resolver is None and FEATURES.get('central_vendor_resolver', False):
        try:
            db_manager.ensure_database_exists()
            conn = sqlite3.connect(DB_PATH, check_same_thread=False)
            _vendor_resolver = VendorResolver(
                db_connection=conn,
                known_mappings=KNOWN_VENDOR_MAPPINGS,
                cache_enabled=FEATURES.get('granular_caching', False)
            )
            logger.info("Global VendorResolver initialized")
        except Exception as e:
            logger.error(f"Failed to initialize global VendorResolver: {e}")
    
    return _vendor_resolver
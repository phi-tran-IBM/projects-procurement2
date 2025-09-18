"""
rag_logic.py - Enhanced RAG Module with Grounded Prompts and Tiered Search
Provides semantic search and natural language understanding for procurement queries
UPDATED: Support for template-based prompts and dual parsing modes
"""

from dotenv import load_dotenv
load_dotenv()

import os
import pandas as pd
import numpy as np
import re
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import hashlib

from langchain_ibm import ChatWatsonx
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationSummaryMemory

# Import discovery store for vector search
from discovery_store import collection

# Import from updated constants (using centralized configuration)
from constants import (
    WATSONX_URL, WATSONX_PROJECT_ID, WATSONX_API_KEY,
    SYNTHESIS_MODEL,  # Using the powerful model for synthesis
    VENDOR_COL, COST_COL, DESC_COL, COMMODITY_COL,
    # NEW: Import dynamic prompt functions
    get_grounded_synthesis_prompt, get_grounded_comparison_prompt,
    get_grounded_recommendation_prompt, get_grounded_statistical_prompt,
    # Import template prompts for direct use if needed
    GROUNDED_SYNTHESIS_PROMPT_TEMPLATE, GROUNDED_COMPARISON_PROMPT_TEMPLATE,
    GROUNDED_RECOMMENDATION_PROMPT_TEMPLATE, GROUNDED_STATISTICAL_PROMPT_TEMPLATE,
    # Import tiered search configuration
    SEMANTIC_SEARCH_TIERS, STRATEGIC_TERM_MAPPINGS,
    # Import quality thresholds
    QUALITY_THRESHOLDS, MIN_DATA_REQUIREMENTS, INSUFFICIENT_DATA_MESSAGES,
    # Features and caching
    FEATURES, CACHE_TTL_BY_TYPE, CACHE_KEY_PREFIXES
)

# Import VendorResolver if available
try:
    from hybrid_rag_architecture import get_vendor_resolver
    VENDOR_RESOLVER_AVAILABLE = True
except ImportError:
    VENDOR_RESOLVER_AVAILABLE = False
    get_vendor_resolver = None

# Import query decomposer for entity extraction (already has unified analysis)
try:
    from query_decomposer import (
        get_decomposer, decompose_query, generate_response,
        QueryIntent, EntityExtraction
    )
    DECOMPOSER_AVAILABLE = True
except ImportError:
    DECOMPOSER_AVAILABLE = False

# Import cache if available
try:
    from simple_cache import QueryCache
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False
    QueryCache = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================
# TEMPLATE RESPONSE EXTRACTION UTILITIES
# ============================================

def extract_template_content(response_text: str) -> str:
    """
    Extract readable content from template-formatted LLM responses.
    Handles various template formats used in the grounded prompts.
    """
    if not response_text or not isinstance(response_text, str):
        return response_text
    
    # Check if template parsing is enabled
    if not FEATURES.get('template_parsing', False):
        return response_text
    
    # Try to extract based on template markers
    
    # For synthesis responses
    if '<ANSWER>' in response_text:
        match = re.search(r'<ANSWER>(.*?)</ANSWER>', response_text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
    
    # For recommendation responses
    if '<RECOMMENDATIONS_START>' in response_text or '<REC1>' in response_text:
        return extract_recommendations_from_template(response_text)
    
    # For comparison responses
    if '<COMPARISON_START>' in response_text or '<VENDOR1>' in response_text:
        return extract_comparison_from_template(response_text)
    
    # For statistical responses
    if '<STATISTICAL_ANALYSIS>' in response_text or '<FINDING1>' in response_text:
        return extract_statistics_from_template(response_text)
    
    # For insufficient data responses
    if '<INSUFFICIENT_DATA>' in response_text:
        match = re.search(r'<INSUFFICIENT_DATA>(.*?)</INSUFFICIENT_DATA>', 
                         response_text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
    
    # Fallback: remove all template tags
    cleaned = re.sub(r'<[^>]+>', '', response_text)
    return cleaned.strip()

def extract_recommendations_from_template(response_text: str) -> str:
    """Extract recommendations from template format"""
    recommendations = []
    
    # Check for insufficient data
    insufficient_match = re.search(r'<INSUFFICIENT_DATA>(.*?)</INSUFFICIENT_DATA>', 
                                 response_text, re.IGNORECASE | re.DOTALL)
    if insufficient_match:
        return insufficient_match.group(1).strip()
    
    # Extract numbered recommendations
    for i in range(1, 11):
        rec_pattern = f'<REC{i}>\\s*<ACTION>(.*?)</ACTION>\\s*<JUSTIFICATION>(.*?)</JUSTIFICATION>\\s*(?:<PRIORITY>(.*?)</PRIORITY>)?\\s*</REC{i}>'
        match = re.search(rec_pattern, response_text, re.IGNORECASE | re.DOTALL)
        if match:
            action = match.group(1).strip()
            justification = match.group(2).strip()
            priority = match.group(3).strip() if match.group(3) else "Medium"
            recommendations.append(f"**{i}. {action}** (Priority: {priority})\n   - {justification}")
    
    if recommendations:
        return "### Strategic Recommendations\n\n" + "\n\n".join(recommendations)
    
    # Fallback
    return re.sub(r'<[^>]+>', '', response_text).strip()

def extract_comparison_from_template(response_text: str) -> str:
    """Extract comparison from template format"""
    result = []
    
    # Extract summary
    summary_match = re.search(r'<SUMMARY>(.*?)</SUMMARY>', response_text, re.IGNORECASE | re.DOTALL)
    if summary_match:
        result.append(f"**Summary:** {summary_match.group(1).strip()}\n")
    
    # Extract vendor details
    for i in range(1, 11):
        vendor_pattern = f'<VENDOR{i}>\\s*<NAME>(.*?)</NAME>\\s*<PERFORMANCE>(.*?)</PERFORMANCE>\\s*(?:<STRENGTHS>(.*?)</STRENGTHS>)?\\s*(?:<CONCERNS>(.*?)</CONCERNS>)?\\s*</VENDOR{i}>'
        match = re.search(vendor_pattern, response_text, re.IGNORECASE | re.DOTALL)
        if match:
            name = match.group(1).strip()
            performance = match.group(2).strip()
            strengths = match.group(3).strip() if match.group(3) else ""
            concerns = match.group(4).strip() if match.group(4) else ""
            
            result.append(f"### {name}")
            result.append(f"- **Performance:** {performance}")
            if strengths:
                result.append(f"- **Strengths:** {strengths}")
            if concerns:
                result.append(f"- **Concerns:** {concerns}")
            result.append("")
    
    # Extract recommendation
    rec_match = re.search(r'<RECOMMENDATION>(.*?)</RECOMMENDATION>', response_text, re.IGNORECASE | re.DOTALL)
    if rec_match:
        result.append(f"**Recommendation:** {rec_match.group(1).strip()}")
    
    if result:
        return "\n".join(result)
    
    return re.sub(r'<[^>]+>', '', response_text).strip()

def extract_statistics_from_template(response_text: str) -> str:
    """Extract statistical analysis from template format"""
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
            findings.append(f"- {match.group(1).strip()}")
    
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
    
    if result:
        return "\n".join(result)
    
    return re.sub(r'<[^>]+>', '', response_text).strip()

# ============================================
# ENHANCED RAG PROCESSOR
# ============================================

class EnhancedRAGProcessor:
    """Advanced RAG processor with grounded prompts and tiered search"""
    
    def __init__(self):
        """Initialize the enhanced RAG processor"""
        try:
            # Initialize LLM with SYNTHESIS_MODEL for high-quality responses
            self.llm = ChatWatsonx(
                model_id=SYNTHESIS_MODEL,
                url=WATSONX_URL,
                project_id=WATSONX_PROJECT_ID or os.getenv("WX_AI_PROJECTID"),
                apikey=WATSONX_API_KEY or os.getenv("WX_AI_APIKEY"),
                params={
                    "decoding_method": "greedy",
                    "max_new_tokens": 800,
                    "temperature": 0.3,
                    "top_p": 0.95,
                    "repetition_penalty": 1.1
                }
            )
            
            # Initialize conversation memory for context
            self.memory = ConversationSummaryMemory(llm=self.llm)
            
            # Initialize decomposer if available (for entity extraction)
            self.decomposer = get_decomposer() if DECOMPOSER_AVAILABLE else None
            
            # Initialize VendorResolver if available
            self.vendor_resolver = get_vendor_resolver() if VENDOR_RESOLVER_AVAILABLE and FEATURES.get('central_vendor_resolver', False) else None
            
            # Initialize search cache if available
            if CACHE_AVAILABLE and FEATURES.get('granular_caching', False):
                self.search_cache = QueryCache(
                    max_size=CACHE_TTL_BY_TYPE.get('semantic_search', 300),
                    ttl_seconds=CACHE_TTL_BY_TYPE.get('semantic_search', 1800)
                )
            else:
                self.search_cache = None
            
            logger.info("Enhanced RAG Processor initialized with template support")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG Processor: {e}")
            raise

    def process_query(self, question: str, mode: str = "semantic") -> Dict[str, Any]:
        """
        Process query with enhanced semantic understanding and grounded responses
        
        Args:
            question: User's query
            mode: Processing mode - "semantic", "hybrid", or "analytical"
        """
        try:
            # Extract entities using decomposer (uses unified analysis if enabled)
            entities = None
            intent = None
            if self.decomposer:
                query_analysis = decompose_query(question)
                entities = query_analysis.get('entities', {})
                intent = query_analysis.get('intent', {})
                
                # Resolve vendor names using VendorResolver
                if self.vendor_resolver and entities.get('vendors'):
                    resolved_vendors = []
                    for vendor in entities['vendors']:
                        resolved = self.vendor_resolver.resolve(vendor)
                        if resolved:
                            resolved_vendors.extend(resolved[:2])  # Top 2 matches
                    entities['vendors'] = list(set(resolved_vendors))
                    logger.info(f"Resolved vendors for semantic search: {entities['vendors']}")
            
            # Perform tiered semantic search
            search_results = self._tiered_semantic_search(question, entities)
            
            if not search_results or len(search_results) == 0:
                return self._create_no_data_response(question, entities)
            
            # Check data quality
            quality_assessment = self._assess_data_quality(search_results)
            if quality_assessment['quality'] == 'low':
                logger.warning(f"Low quality data for query: {question[:50]}...")
            
            # Build enhanced context
            context = self._build_enhanced_context(search_results, entities, intent, quality_assessment)
            
            # Generate response using appropriate grounded prompt with template support
            response = self._generate_grounded_response(question, context, intent, search_results)
            
            # Calculate confidence score
            confidence = self._calculate_confidence(search_results, quality_assessment)
            
            return {
                "summary": response,
                "answer": response,
                "records_analyzed": len(search_results),
                "confidence": confidence,
                "source": "RAG",
                "search_tier_used": quality_assessment.get('search_tier', 'unknown'),
                "data_quality": quality_assessment['quality'],
                "entities_identified": entities if entities else {},
                "grounded_prompts_used": FEATURES.get('grounded_prompts', False),
                "template_parsing_used": FEATURES.get('template_parsing', False)
            }
            
        except Exception as e:
            logger.error(f"RAG processing failed: {e}")
            return {
                "error": str(e),
                "summary": "An error occurred during semantic processing.",
                "answer": f"I encountered an error while processing your query: {str(e)}",
                "confidence": 0,
                "source": "RAG"
            }

    def _tiered_semantic_search(self, question: str, entities: Optional[Dict] = None) -> List[Dict]:
        """
        Perform tiered semantic search with progressive expansion.
        Implements SEMANTIC_SEARCH_TIERS from constants.
        """
        # Check cache first
        cache_key = hashlib.md5(f"{question}{entities}".encode()).hexdigest()
        if self.search_cache:
            cached = self.search_cache.get(f"{CACHE_KEY_PREFIXES['semantic']}{cache_key}")
            if cached:
                logger.info("Semantic search cache hit")
                return cached
        
        results = []
        search_tier_used = None
        
        # Try each tier progressively
        for tier_name, tier_config in SEMANTIC_SEARCH_TIERS.items():
            logger.info(f"Trying semantic search {tier_name}: {tier_config['description']}")
            
            # Modify query based on tier
            search_query = self._modify_query_for_tier(question, entities, tier_config)
            
            # Perform search
            try:
                search_results = collection.query(
                    query_texts=[search_query],
                    n_results=tier_config['n_results']
                )
                
                if search_results and search_results['metadatas'] and search_results['metadatas'][0]:
                    # Filter by relevance threshold
                    filtered_results = self._filter_by_relevance(
                        search_results,
                        tier_config['min_relevance']
                    )
                    
                    if filtered_results and len(filtered_results) >= 5:  # Need at least 5 results
                        results = filtered_results
                        search_tier_used = tier_name
                        logger.info(f"Found {len(results)} results in {tier_name}")
                        break
                    else:
                        logger.info(f"Insufficient results in {tier_name}, trying next tier")
                
            except Exception as e:
                logger.error(f"Search failed for {tier_name}: {e}")
                continue
        
        # Process and rank results
        if results:
            results = self._process_and_rank_results(results, entities)
            
            # Cache the results
            if self.search_cache:
                self.search_cache.set(f"{CACHE_KEY_PREFIXES['semantic']}{cache_key}", results)
            
            # Store which tier was used
            if results and len(results) > 0:
                results[0]['search_tier'] = search_tier_used
        
        return results

    def _modify_query_for_tier(self, question: str, entities: Optional[Dict], tier_config: Dict) -> str:
        """
        Modify query based on tier requirements.
        Uses STRATEGIC_TERM_MAPPINGS from constants.
        """
        if not tier_config.get('query_modification'):
            return question  # Tier 1: exact query
        
        modified_query = question
        
        if tier_config['query_modification'] == 'add_synonyms':
            # Tier 2: Add synonyms and related terms
            expanded_terms = []
            
            # Check strategic term mappings
            question_lower = question.lower()
            for strategic_term, synonyms in STRATEGIC_TERM_MAPPINGS.items():
                if strategic_term in question_lower:
                    expanded_terms.extend(synonyms)
            
            # Add entity context
            if entities:
                if entities.get('vendors'):
                    expanded_terms.extend([f"vendor:{v}" for v in entities['vendors'][:3]])
                if entities.get('metrics'):
                    expanded_terms.extend([f"metric:{m}" for m in entities['metrics'][:3]])
                if entities.get('commodities'):
                    expanded_terms.extend([f"category:{c}" for c in entities['commodities'][:3]])
            
            if expanded_terms:
                modified_query = f"{question} {' '.join(expanded_terms)}"
        
        elif tier_config['query_modification'] == 'category_search':
            # Tier 3: Broad category search
            categories = []
            
            # Extract broad categories from question
            if 'vendor' in question.lower() or 'supplier' in question.lower():
                categories.append('vendor management procurement suppliers')
            if 'cost' in question.lower() or 'spend' in question.lower():
                categories.append('cost spending budget financial')
            if 'optimization' in question.lower() or 'improve' in question.lower():
                categories.append('optimization improvement efficiency savings')
            
            if categories:
                modified_query = f"{question} {' '.join(categories)}"
            else:
                # Fallback to very broad search
                modified_query = f"{question} procurement spending vendors"
        
        logger.debug(f"Modified query for tier: {modified_query[:100]}...")
        return modified_query

    def _filter_by_relevance(self, search_results: Dict, min_relevance: float) -> List[Dict]:
        """Filter search results by relevance score"""
        filtered = []
        
        if not search_results.get('metadatas') or not search_results['metadatas'][0]:
            return filtered
        
        metadatas = search_results['metadatas'][0]
        distances = search_results.get('distances', [[]])[0] if 'distances' in search_results else []
        
        for i, metadata in enumerate(metadatas):
            # Calculate relevance from distance (lower distance = higher relevance)
            if i < len(distances):
                distance = distances[i]
                relevance = 1 / (1 + distance)  # Convert distance to similarity
            else:
                relevance = 0.5  # Default if no distance available
            
            if relevance >= min_relevance:
                metadata['relevance_score'] = relevance
                filtered.append(metadata)
        
        return filtered

    def _process_and_rank_results(self, results: List[Dict], entities: Optional[Dict]) -> List[Dict]:
        """
        Process and rank results with entity boost and vendor resolution.
        """
        processed = []
        
        for result in results:
            # Boost relevance for entity matches
            relevance_boost = 1.0
            
            if entities and result:
                # Check vendor matches (with resolved names)
                if entities.get('vendors') and VENDOR_COL in result:
                    vendor_field = result.get(VENDOR_COL, '').upper()
                    for vendor in entities['vendors']:
                        if vendor.upper() in vendor_field:
                            relevance_boost *= 1.5
                            logger.debug(f"Boosted relevance for vendor match: {vendor}")
                
                # Check commodity matches
                if entities.get('commodities') and COMMODITY_COL in result:
                    commodity_field = result.get(COMMODITY_COL, '').upper()
                    for commodity in entities['commodities']:
                        if commodity.upper() in commodity_field:
                            relevance_boost *= 1.3
            
            # Apply boost
            original_relevance = result.get('relevance_score', 0.5)
            result['relevance_score'] = min(1.0, original_relevance * relevance_boost)
            processed.append(result)
        
        # Sort by relevance
        processed.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        # Return top results
        return processed[:50]

    def _assess_data_quality(self, search_results: List[Dict]) -> Dict[str, Any]:
        """
        Assess the quality of search results.
        Uses QUALITY_THRESHOLDS from constants.
        """
        if not search_results:
            return {'quality': 'none', 'details': 'No results found'}
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(search_results)
        
        # Calculate metrics
        num_results = len(df)
        avg_relevance = np.mean([r.get('relevance_score', 0) for r in search_results[:10]])
        
        # Check for nulls in key columns
        null_percentage = 0
        key_columns = [VENDOR_COL, COST_COL, DESC_COL]
        for col in key_columns:
            if col in df.columns:
                null_percentage += df[col].isna().sum() / len(df)
        null_percentage = null_percentage / len(key_columns) if key_columns else 0
        
        # Determine quality level
        quality = 'low'
        for level, thresholds in QUALITY_THRESHOLDS.items():
            if (num_results >= thresholds['min_data_points'] and
                avg_relevance >= thresholds['min_relevance'] and
                null_percentage <= thresholds['max_null_percentage']):
                quality = level.replace('_quality', '')
                break
        
        # Get search tier if available
        search_tier = search_results[0].get('search_tier', 'unknown') if search_results else 'none'
        
        return {
            'quality': quality,
            'num_results': num_results,
            'avg_relevance': float(avg_relevance),
            'null_percentage': float(null_percentage),
            'search_tier': search_tier
        }

    def _build_enhanced_context(self, search_results: List[Dict], 
                               entities: Optional[Dict],
                               intent: Optional[Dict],
                               quality_assessment: Dict) -> str:
        """
        Build comprehensive context from search results with data quality indicators.
        Enhanced version that provides better structure for grounded prompts.
        """
        if not search_results:
            return "No relevant data found."
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(search_results)
        
        context_parts = []
        
        # Data quality indicator
        context_parts.append(f"Data Quality: {quality_assessment['quality'].upper()}")
        context_parts.append(f"Records Found: {len(df)}")
        context_parts.append(f"Average Relevance: {quality_assessment['avg_relevance']:.2%}\n")
        
        # Vendor analysis with resolved names
        if VENDOR_COL in df.columns:
            vendor_counts = df[VENDOR_COL].value_counts()
            top_vendors = vendor_counts.head(5)
            
            if not top_vendors.empty:
                context_parts.append("**Top Vendors in Results:**")
                for vendor, count in top_vendors.items():
                    context_parts.append(f"  - {vendor}: {count} records")
        
        # Cost analysis
        if COST_COL in df.columns:
            # Clean and convert cost data
            df['COST_NUMERIC'] = pd.to_numeric(
                df[COST_COL].astype(str).str.replace(',', '').str.replace('$', ''),
                errors='coerce'
            )
            
            valid_costs = df['COST_NUMERIC'].dropna()
            if not valid_costs.empty:
                context_parts.append("\n**Financial Metrics:**")
                context_parts.append(f"  - Total Value: ${valid_costs.sum():,.2f}")
                context_parts.append(f"  - Average: ${valid_costs.mean():,.2f}")
                context_parts.append(f"  - Median: ${valid_costs.median():,.2f}")
                context_parts.append(f"  - Range: ${valid_costs.min():,.2f} - ${valid_costs.max():,.2f}")
        
        # Category analysis
        if COMMODITY_COL in df.columns:
            categories = df[COMMODITY_COL].value_counts().head(5)
            if not categories.empty:
                context_parts.append("\n**Main Categories:**")
                for cat, count in categories.items():
                    context_parts.append(f"  - {cat}: {count} records")
        
        # Sample records for context
        context_parts.append("\n**Sample Records (Top 5 by Relevance):**")
        for i, row in enumerate(df.head(5).itertuples(), 1):
            record_parts = []
            
            if hasattr(row, VENDOR_COL.replace(' ', '_')):
                vendor = getattr(row, VENDOR_COL.replace(' ', '_'))
                record_parts.append(f"Vendor: {vendor}")
            
            if hasattr(row, COST_COL.replace(' ', '_')):
                cost = getattr(row, COST_COL.replace(' ', '_'))
                record_parts.append(f"Cost: {cost}")
            
            if hasattr(row, DESC_COL.replace(' ', '_')):
                desc = str(getattr(row, DESC_COL.replace(' ', '_')))[:100]
                record_parts.append(f"Description: {desc}...")
            
            context_parts.append(f"{i}. {' | '.join(record_parts)}")
        
        # Add entity context if available
        if entities:
            context_parts.append("\n**Query Context:**")
            if entities.get('vendors'):
                context_parts.append(f"  - Vendors of Interest: {', '.join(entities['vendors'])}")
            if entities.get('metrics'):
                context_parts.append(f"  - Metrics Requested: {', '.join(entities['metrics'])}")
        
        return '\n'.join(context_parts)

    def _generate_grounded_response(self, question: str, context: str, 
                                   intent: Optional[Dict], 
                                   search_results: List[Dict]) -> str:
        """
        Generate response using appropriate grounded prompt from constants.
        UPDATED: Now uses dynamic prompt functions with template support.
        """
        # Select the appropriate grounded prompt based on intent
        if FEATURES.get('grounded_prompts', False) and intent:
            intent_type = intent.get('primary_intent', 'other')
            
            if intent_type == 'comparison':
                # Use dynamic comparison prompt
                vendor_data = self._extract_vendor_data(search_results)
                prompt = get_grounded_comparison_prompt().format(
                    vendor_data=json.dumps(vendor_data, indent=2),
                    question=question
                )
                
            elif intent_type == 'recommendation':
                # Use dynamic recommendation prompt
                prompt = get_grounded_recommendation_prompt().format(
                    context=context,
                    focus=question,
                    question=question
                )
                
            elif intent_type == 'statistical':
                # Use dynamic statistical prompt
                statistics = self._calculate_statistics(search_results)
                prompt = get_grounded_statistical_prompt().format(
                    statistics=json.dumps(statistics, indent=2),
                    question=question
                )
                
            else:
                # Use general dynamic synthesis prompt
                prompt = get_grounded_synthesis_prompt().format(
                    context=context,
                    question=question
                )
        else:
            # Fallback to basic grounded prompt if features disabled
            prompt = f"""Based ONLY on the following data, answer the question.
If the data doesn't support an answer, say "I don't have sufficient data."

Data Context:
{context}

Question: {question}

Answer:"""
        
        # Generate response with LLM
        try:
            chain = LLMChain(llm=self.llm, prompt=PromptTemplate(template=prompt, input_variables=[]))
            response = chain.run()
            
            # Extract content from template if template parsing is enabled
            if FEATURES.get('template_parsing', False):
                response = extract_template_content(response)
            
            return response.strip()
        except Exception as e:
            logger.error(f"LLM response generation failed: {e}")
            # Fallback to basic summary
            return self._generate_fallback_response(context, search_results)

    def _extract_vendor_data(self, search_results: List[Dict]) -> List[Dict]:
        """Extract vendor-specific data for comparison prompt"""
        vendor_data = {}
        
        for result in search_results:
            vendor = result.get(VENDOR_COL)
            if vendor:
                if vendor not in vendor_data:
                    vendor_data[vendor] = {
                        'vendor': vendor,
                        'records': 0,
                        'total_cost': 0,
                        'costs': []
                    }
                
                vendor_data[vendor]['records'] += 1
                
                # Try to extract cost
                cost_str = result.get(COST_COL, '')
                try:
                    cost = float(str(cost_str).replace(',', '').replace('$', ''))
                    vendor_data[vendor]['costs'].append(cost)
                    vendor_data[vendor]['total_cost'] += cost
                except:
                    pass
        
        # Calculate averages
        for vendor in vendor_data.values():
            if vendor['costs']:
                vendor['avg_cost'] = sum(vendor['costs']) / len(vendor['costs'])
                vendor['min_cost'] = min(vendor['costs'])
                vendor['max_cost'] = max(vendor['costs'])
                del vendor['costs']  # Remove raw data
        
        return list(vendor_data.values())

    def _calculate_statistics(self, search_results: List[Dict]) -> Dict[str, Any]:
        """Calculate statistics from search results"""
        df = pd.DataFrame(search_results)
        stats = {'record_count': len(df)}
        
        if COST_COL in df.columns:
            # Extract numeric costs
            costs = []
            for cost_str in df[COST_COL]:
                try:
                    cost = float(str(cost_str).replace(',', '').replace('$', ''))
                    costs.append(cost)
                except:
                    pass
            
            if costs:
                stats['cost_statistics'] = {
                    'count': len(costs),
                    'total': sum(costs),
                    'mean': np.mean(costs),
                    'median': np.median(costs),
                    'std': np.std(costs),
                    'min': min(costs),
                    'max': max(costs)
                }
        
        return stats

    def _generate_fallback_response(self, context: str, search_results: List[Dict]) -> str:
        """Generate basic response without LLM"""
        if not search_results:
            return "No relevant data found to answer your question."
        
        response_parts = [f"Found {len(search_results)} relevant records.\n"]
        
        # Add basic summary from context
        if "Total Value:" in context:
            # Extract financial metrics
            for line in context.split('\n'):
                if 'Total Value:' in line or 'Average:' in line:
                    response_parts.append(line.strip())
        
        return '\n'.join(response_parts)

    def _create_no_data_response(self, question: str, entities: Optional[Dict]) -> Dict[str, Any]:
        """
        Create response when no data is found.
        Uses INSUFFICIENT_DATA_MESSAGES from constants.
        """
        # Try to provide helpful context
        if entities and entities.get('vendors'):
            # Check if vendor might be misspelled
            vendor = entities['vendors'][0]
            message = INSUFFICIENT_DATA_MESSAGES['vendor_not_found'].format(
                vendor=vendor,
                suggestions="Try checking the spelling or using a different vendor name."
            )
        else:
            message = INSUFFICIENT_DATA_MESSAGES['no_data']
        
        return {
            "summary": message,
            "answer": message,
            "records_analyzed": 0,
            "confidence": 10,
            "source": "RAG",
            "data_quality": "none"
        }

    def _calculate_confidence(self, search_results: List[Dict], quality_assessment: Dict) -> int:
        """Calculate confidence score based on results and quality"""
        confidence = 50  # Base confidence
        
        # Adjust based on data quality
        quality_map = {'high': 30, 'medium': 20, 'low': 10, 'none': 0}
        confidence += quality_map.get(quality_assessment['quality'], 0)
        
        # Adjust based on number of results
        if len(search_results) > 20:
            confidence += 20
        elif len(search_results) > 10:
            confidence += 10
        elif len(search_results) < 5:
            confidence -= 20
        
        # Adjust based on average relevance
        avg_relevance = quality_assessment.get('avg_relevance', 0.5)
        confidence += int(avg_relevance * 20)
        
        # Adjust based on search tier used
        tier_penalty = {'tier_1_exact': 0, 'tier_2_expanded': -5, 'tier_3_broad': -10}
        search_tier = quality_assessment.get('search_tier', 'unknown')
        confidence += tier_penalty.get(search_tier, -5)
        
        # Cap between 0 and 100
        return max(0, min(100, confidence))

# ============================================
# SINGLETON INSTANCE
# ============================================

_rag_processor = None

def get_rag_processor() -> EnhancedRAGProcessor:
    """Get singleton RAG processor instance"""
    global _rag_processor
    if _rag_processor is None:
        _rag_processor = EnhancedRAGProcessor()
    return _rag_processor

# ============================================
# PUBLIC INTERFACE (Backward Compatible)
# ============================================

def answer_question_intelligent(question: str, mode: str = "semantic") -> Dict[str, Any]:
    """
    Main entry point for RAG processing (backward compatible)
    Now uses grounded prompts with template support
    """
    try:
        processor = get_rag_processor()
        return processor.process_query(question, mode)
    except Exception as e:
        logger.error(f"Failed to process question: {e}")
        return {
            "error": str(e),
            "summary": "An error occurred during processing.",
            "answer": f"I encountered an error: {str(e)}",
            "confidence": 0,
            "source": "RAG"
        }

def analyze_vendor_semantic(vendor_name: str) -> Dict[str, Any]:
    """
    Perform semantic analysis for a specific vendor
    Now uses VendorResolver for better matching
    """
    # Resolve vendor name first if VendorResolver available
    if VENDOR_RESOLVER_AVAILABLE and FEATURES.get('central_vendor_resolver', False):
        resolver = get_vendor_resolver()
        if resolver:
            resolved = resolver.get_canonical_name(vendor_name)
            if resolved:
                vendor_name = resolved
                logger.info(f"Resolved vendor name to: {vendor_name}")
    
    question = f"Provide a comprehensive analysis of {vendor_name} including spending patterns, categories, and relationships"
    return answer_question_intelligent(question, mode="analytical")

def get_recommendations(context: str = "cost optimization") -> Dict[str, Any]:
    """
    Get strategic recommendations based on semantic analysis
    Now uses dynamic GROUNDED_RECOMMENDATION_PROMPT with template support
    """
    question = f"What are the top recommendations for {context} based on procurement patterns?"
    result = answer_question_intelligent(question, mode="semantic")
    
    # Mark as using grounded prompts and template parsing
    result['recommendation_type'] = context
    result['grounded_recommendation'] = FEATURES.get('grounded_prompts', False)
    result['template_parsing'] = FEATURES.get('template_parsing', False)
    
    return result

def compare_vendors_semantic(vendors: List[str]) -> Dict[str, Any]:
    """
    Perform semantic comparison of multiple vendors
    Now uses VendorResolver and dynamic GROUNDED_COMPARISON_PROMPT with template support
    """
    # Resolve vendor names if VendorResolver available
    if VENDOR_RESOLVER_AVAILABLE and FEATURES.get('central_vendor_resolver', False):
        resolver = get_vendor_resolver()
        if resolver:
            resolved_vendors = []
            for vendor in vendors:
                resolved = resolver.get_canonical_name(vendor)
                resolved_vendors.append(resolved if resolved else vendor)
            vendors = resolved_vendors
            logger.info(f"Resolved vendor names: {vendors}")
    
    vendor_str = " and ".join(vendors)
    question = f"Compare {vendor_str} across all dimensions including spending, categories, and performance"
    
    result = answer_question_intelligent(question, mode="semantic")
    result['comparison_vendors'] = vendors
    result['template_parsing'] = FEATURES.get('template_parsing', False)
    
    return result

# ============================================
# TESTING
# ============================================

if __name__ == "__main__":
    # Test the enhanced RAG processor with template support
    test_queries = [
        "What are the spending patterns for Dell?",
        "Compare Microsoft and IBM procurement strategies",
        "Which vendors should we optimize for cost reduction?",
        "Analyze technology procurement trends",
        "What's the relationship between order size and vendor performance?",
    ]
    
    print("Testing Enhanced RAG Processor with Template Support")
    print("=" * 60)
    
    # Enable all features for testing
    FEATURES['grounded_prompts'] = True
    FEATURES['central_vendor_resolver'] = True
    FEATURES['tiered_search'] = True
    FEATURES['template_parsing'] = True
    
    processor = get_rag_processor()
    
    for query in test_queries[:2]:  # Test first 2 queries
        print(f"\nQuery: {query}")
        print("-" * 40)
        
        result = processor.process_query(query)
        
        print(f"Confidence: {result.get('confidence')}%")
        print(f"Records Analyzed: {result.get('records_analyzed')}")
        print(f"Data Quality: {result.get('data_quality', 'unknown')}")
        print(f"Search Tier Used: {result.get('search_tier_used', 'unknown')}")
        print(f"Grounded Prompts: {result.get('grounded_prompts_used', False)}")
        print(f"Template Parsing: {result.get('template_parsing_used', False)}")
        
        if result.get('entities_identified'):
            entities = result['entities_identified']
            if entities.get('vendors'):
                print(f"Vendors Identified: {entities['vendors']}")
        
        print(f"\nAnswer Preview:")
        answer = result.get('answer', 'No answer')
        print(answer[:500] + "..." if len(answer) > 500 else answer)
        print("-" * 40)
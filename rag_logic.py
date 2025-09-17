"""
rag_logic.py - Enhanced RAG Module with Advanced Prompt Engineering
Provides semantic search and natural language understanding for procurement queries
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

# --- MODIFIED: Import new specialized model constant ---
from constants import (
    WATSONX_URL, WATSONX_PROJECT_ID, WATSONX_API_KEY,
    SYNTHESIS_MODEL,  # Changed from LLM_MODEL
    VENDOR_COL, COST_COL, DESC_COL, COMMODITY_COL
)

# ============================================
# IMPORT QUERY DECOMPOSER
# ============================================
try:
    from query_decomposer import (
        get_decomposer, decompose_query, generate_response,
        QueryIntent, EntityExtraction
    )
    DECOMPOSER_AVAILABLE = True
except ImportError:
    DECOMPOSER_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================
# ENHANCED RAG PROCESSOR
# ============================================

class EnhancedRAGProcessor:
    """Advanced RAG processor with improved prompt engineering and semantic understanding"""
    
    def __init__(self):
        """Initialize the enhanced RAG processor"""
        try:
            # --- MODIFIED: Initialize LLM with the powerful SYNTHESIS_MODEL ---
            self.llm = ChatWatsonx(
                model_id=SYNTHESIS_MODEL,
                url=WATSONX_URL,
                project_id=WATSONX_PROJECT_ID or os.getenv("WX_AI_PROJECTID"),
                apikey=WATSONX_API_KEY or os.getenv("WX_AI_APIKEY"),
                params={
                    "decoding_method": "greedy",
                    "max_new_tokens": 800,
                    "temperature": 0.3,  # Slightly higher for more natural responses
                    "top_p": 0.95,
                    "repetition_penalty": 1.1
                }
            )
            
            # Initialize conversation memory for context
            self.memory = ConversationSummaryMemory(llm=self.llm)
            
            # Initialize decomposer if available
            self.decomposer = get_decomposer() if DECOMPOSER_AVAILABLE else None
            
            # Cache for semantic search results
            self._search_cache = {}
            
            logger.info("Enhanced RAG Processor initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG Processor: {e}")
            raise

    def process_query(self, question: str, mode: str = "semantic") -> Dict[str, Any]:
        """
        Process query with enhanced semantic understanding
        
        Args:
            question: User's query
            mode: Processing mode - "semantic", "hybrid", or "analytical"
        """
        try:
            # Extract entities if decomposer available
            entities = None
            intent = None
            if self.decomposer:
                query_analysis = decompose_query(question)
                entities = query_analysis.get('entities', {})
                intent = query_analysis.get('intent', {})
            
            # Perform semantic search
            search_results = self._enhanced_semantic_search(question, entities)
            
            if not search_results or len(search_results) == 0:
                return {
                    "summary": "No relevant data found in the knowledge base.",
                    "answer": "I couldn't find specific information about that in the procurement data. Please try rephrasing your question or asking about specific vendors or categories.",
                    "confidence": 10,
                    "source": "RAG"
                }
            
            # Build context from search results
            context = self._build_enhanced_context(search_results, entities, intent)
            
            # Generate response based on query type
            if intent and intent.get('primary_intent') == 'recommendation':
                response = self._generate_recommendation_response(question, context, search_results)
            elif intent and intent.get('primary_intent') == 'comparison':
                response = self._generate_comparison_response(question, context, search_results, entities)
            elif mode == "analytical":
                response = self._generate_analytical_response(question, context, search_results)
            else:
                response = self._generate_semantic_response(question, context, search_results)
            
            # Calculate confidence score
            confidence = self._calculate_confidence(search_results, response)
            
            return {
                "summary": response,
                "answer": response,
                "records_analyzed": len(search_results),
                "confidence": confidence,
                "source": "RAG",
                "search_relevance": self._calculate_relevance_score(search_results),
                "entities_identified": entities if entities else {}
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

    def _enhanced_semantic_search(self, question: str, entities: Optional[Dict] = None) -> List[Dict]:
        """
        Perform enhanced semantic search with entity awareness
        """
        # Check cache
        cache_key = hashlib.md5(question.encode()).hexdigest()
        if cache_key in self._search_cache:
            logger.info("Using cached search results")
            return self._search_cache[cache_key]
        
        try:
            # Build enhanced query
            enhanced_query = question
            
            # Add entity context to improve search
            if entities:
                if entities.get('vendors'):
                    enhanced_query += f" vendors: {' '.join(entities['vendors'][:3])}"
                if entities.get('commodities'):
                    enhanced_query += f" categories: {' '.join(entities['commodities'][:3])}"
                if entities.get('metrics'):
                    enhanced_query += f" metrics: {' '.join(entities['metrics'][:3])}"
            
            # Perform vector search
            results = collection.query(
                query_texts=[enhanced_query],
                n_results=50  # Get more results for better context
            )
            
            if not results['metadatas'][0]:
                # Try simplified query if enhanced query returns nothing
                results = collection.query(
                    query_texts=[question],
                    n_results=30
                )
            
            # Process and rank results
            processed_results = self._process_search_results(results, entities)
            
            # Cache results
            self._search_cache[cache_key] = processed_results
            
            return processed_results
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []

    def _process_search_results(self, results: Dict, entities: Optional[Dict] = None) -> List[Dict]:
        """
        Process and rank search results based on relevance
        """
        if not results.get('metadatas') or not results['metadatas'][0]:
            return []
        
        processed = []
        
        for i, metadata in enumerate(results['metadatas'][0]):
            # Calculate relevance score
            relevance = 1.0
            
            # Boost relevance for entity matches
            if entities and metadata:
                if entities.get('vendors'):
                    vendor_field = metadata.get(VENDOR_COL, '').upper()
                    for vendor in entities['vendors']:
                        if vendor.upper() in vendor_field:
                            relevance *= 1.5
                
                if entities.get('commodities'):
                    commodity_field = metadata.get(COMMODITY_COL, '').upper()
                    for commodity in entities['commodities']:
                        if commodity.upper() in commodity_field:
                            relevance *= 1.3
            
            # Add distance score if available
            if results.get('distances') and results['distances'][0]:
                distance = results['distances'][0][i]
                # Convert distance to similarity (lower distance = higher similarity)
                similarity = 1 / (1 + distance)
                relevance *= similarity
            
            metadata['relevance_score'] = relevance
            processed.append(metadata)
        
        # Sort by relevance
        processed.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        return processed[:30]  # Return top 30 most relevant

    def _build_enhanced_context(self, search_results: List[Dict], 
                               entities: Optional[Dict] = None,
                               intent: Optional[Dict] = None) -> str:
        """
        Build comprehensive context from search results
        """
        if not search_results:
            return "No relevant data found."
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(search_results)
        
        context_parts = []
        
        # Overall statistics
        context_parts.append(f"Analyzing {len(df)} relevant procurement records.")
        
        # Vendor analysis
        if VENDOR_COL in df.columns:
            vendor_counts = df[VENDOR_COL].value_counts()
            top_vendors = vendor_counts.head(5)
            
            if not top_vendors.empty:
                vendor_list = [f"{v} ({c} records)" for v, c in top_vendors.items()]
                context_parts.append(f"Top vendors in results: {', '.join(vendor_list)}")
        
        # Cost analysis
        if COST_COL in df.columns:
            # Clean and convert cost data
            df['COST_NUMERIC'] = pd.to_numeric(
                df[COST_COL].astype(str).str.replace(',', '').str.replace('$', ''),
                errors='coerce'
            )
            
            valid_costs = df['COST_NUMERIC'].dropna()
            if not valid_costs.empty:
                total_cost = valid_costs.sum()
                avg_cost = valid_costs.mean()
                median_cost = valid_costs.median()
                
                context_parts.append(f"Financial metrics from search results:")
                context_parts.append(f"  - Total value: ${total_cost:,.2f}")
                context_parts.append(f"  - Average: ${avg_cost:,.2f}")
                context_parts.append(f"  - Median: ${median_cost:,.2f}")
        
        # Category analysis
        if COMMODITY_COL in df.columns:
            categories = df[COMMODITY_COL].value_counts().head(5)
            if not categories.empty:
                cat_list = [f"{cat} ({count})" for cat, count in categories.items()]
                context_parts.append(f"Main categories: {', '.join(cat_list)}")
        
        # Add intent-specific context
        if intent:
            intent_type = intent.get('primary_intent', '')
            if intent_type == 'comparison' and entities and entities.get('vendors'):
                # Focus on comparison data
                vendor_data = {}
                for vendor in entities['vendors']:
                    vendor_df = df[df[VENDOR_COL].str.upper().str.contains(vendor.upper(), na=False)]
                    if not vendor_df.empty:
                        vendor_data[vendor] = {
                            'count': len(vendor_df),
                            'total': vendor_df['COST_NUMERIC'].sum() if 'COST_NUMERIC' in vendor_df else 0,
                            'avg': vendor_df['COST_NUMERIC'].mean() if 'COST_NUMERIC' in vendor_df else 0
                        }
                
                if vendor_data:
                    context_parts.append("\nVendor-specific data from search:")
                    for vendor, data in vendor_data.items():
                        context_parts.append(f"  {vendor}: {data['count']} records, "
                                           f"${data['total']:,.2f} total, "
                                           f"${data['avg']:,.2f} average")
        
        # Sample records for detailed context
        context_parts.append("\nSample records (top 5 by relevance):")
        for i, row in enumerate(df.head(5).itertuples(), 1):
            record_desc = f"{i}. "
            if hasattr(row, VENDOR_COL.replace(' ', '_')):
                record_desc += f"Vendor: {getattr(row, VENDOR_COL.replace(' ', '_'))}, "
            if hasattr(row, COST_COL.replace(' ', '_')):
                record_desc += f"Cost: {getattr(row, COST_COL.replace(' ', '_'))}, "
            if hasattr(row, DESC_COL.replace(' ', '_')):
                desc = str(getattr(row, DESC_COL.replace(' ', '_')))[:100]
                record_desc += f"Description: {desc}..."
            context_parts.append(record_desc)
        
        return '\n'.join(context_parts)

    def _generate_semantic_response(self, question: str, context: str, 
                                   search_results: List[Dict]) -> str:
        """
        Generate standard semantic response with improved prompting
        """
        prompt = PromptTemplate(
            template="""You are an expert procurement analyst. Answer the following question based on the search results from the procurement database.

Question: {question}

Context from search results:
{context}

Guidelines:
1. Provide a clear, concise, and informative answer
2. Use specific numbers and examples from the data when available
3. If the data doesn't fully answer the question, acknowledge limitations
4. Format numbers with proper currency formatting (e.g., $1,234.56)
5. Be professional but conversational
6. If multiple interpretations exist, address the most likely one
7. Highlight key insights or patterns you notice

Answer:""",
            input_variables=["question", "context"]
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        response = chain.run(question=question, context=context)
        
        return response.strip()

    def _generate_comparison_response(self, question: str, context: str,
                                     search_results: List[Dict],
                                     entities: Optional[Dict] = None) -> str:
        """
        Generate comparison-focused response
        """
        # Extract vendor-specific data
        vendor_analysis = self._analyze_vendors_from_results(search_results, entities)
        
        prompt = PromptTemplate(
            template="""You are an expert procurement analyst specializing in vendor comparisons. 
Analyze and compare the vendors based on the search results.

Question: {question}

Context from search results:
{context}

Vendor-specific analysis:
{vendor_analysis}

Guidelines for comparison:
1. Clearly compare the vendors on multiple dimensions (cost, volume, categories)
2. Highlight key differences and similarities
3. Provide specific metrics for each vendor
4. Identify which vendor performs better in different aspects
5. Offer insights about vendor relationships
6. Use tables or bullet points for clarity
7. Conclude with a summary of findings

Comparison Analysis:""",
            input_variables=["question", "context", "vendor_analysis"]
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        response = chain.run(
            question=question,
            context=context,
            vendor_analysis=json.dumps(vendor_analysis, indent=2)
        )
        
        return response.strip()

    def _generate_recommendation_response(self, question: str, context: str,
                                        search_results: List[Dict]) -> str:
        """
        Generate recommendation-focused response
        """
        # Analyze patterns for recommendations
        patterns = self._identify_patterns(search_results)
        
        prompt = PromptTemplate(
            template="""You are a strategic procurement advisor. Based on the search results, 
provide actionable recommendations.

Question: {question}

Context from search results:
{context}

Identified patterns:
{patterns}

Guidelines for recommendations:
1. Provide clear, actionable recommendations
2. Base recommendations on data patterns and insights
3. Consider cost optimization opportunities
4. Identify risks and opportunities
5. Suggest specific actions with expected outcomes
6. Prioritize recommendations by potential impact
7. Include metrics to track success
8. Consider both short-term and long-term strategies

Strategic Recommendations:""",
            input_variables=["question", "context", "patterns"]
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        response = chain.run(
            question=question,
            context=context,
            patterns=json.dumps(patterns, indent=2)
        )
        
        return response.strip()

    def _generate_analytical_response(self, question: str, context: str,
                                     search_results: List[Dict]) -> str:
        """
        Generate deep analytical response
        """
        # Perform statistical analysis
        stats = self._calculate_statistics(search_results)
        
        prompt = PromptTemplate(
            template="""You are a data analyst specializing in procurement analytics. 
Provide a comprehensive analysis based on the search results.

Question: {question}

Context from search results:
{context}

Statistical analysis:
{statistics}

Guidelines for analysis:
1. Provide thorough statistical analysis
2. Identify trends and patterns
3. Explain correlations and relationships
4. Use visualizations descriptions (e.g., "if plotted, this would show...")
5. Discuss outliers and anomalies
6. Provide confidence levels for findings
7. Suggest areas for further investigation
8. Include methodology notes where relevant

Analytical Report:""",
            input_variables=["question", "context", "statistics"]
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        response = chain.run(
            question=question,
            context=context,
            statistics=json.dumps(stats, indent=2)
        )
        
        return response.strip()

    def _analyze_vendors_from_results(self, search_results: List[Dict],
                                     entities: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Analyze vendor-specific data from search results
        """
        if not search_results:
            return {}
        
        df = pd.DataFrame(search_results)
        vendor_analysis = {}
        
        if VENDOR_COL not in df.columns:
            return vendor_analysis
        
        # Get list of vendors to analyze
        vendors_to_analyze = []
        if entities and entities.get('vendors'):
            vendors_to_analyze = entities['vendors']
        else:
            # Get top vendors from results
            vendor_counts = df[VENDOR_COL].value_counts()
            vendors_to_analyze = vendor_counts.head(5).index.tolist()
        
        # Analyze each vendor
        for vendor in vendors_to_analyze:
            vendor_df = df[df[VENDOR_COL].str.upper().str.contains(vendor.upper(), na=False)]
            
            if vendor_df.empty:
                continue
            
            analysis = {
                'record_count': len(vendor_df),
                'categories': vendor_df[COMMODITY_COL].value_counts().head(3).to_dict() 
                             if COMMODITY_COL in vendor_df else {},
            }
            
            # Cost analysis
            if COST_COL in vendor_df.columns:
                vendor_df['COST_NUMERIC'] = pd.to_numeric(
                    vendor_df[COST_COL].astype(str).str.replace(',', '').str.replace('$', ''),
                    errors='coerce'
                )
                
                valid_costs = vendor_df['COST_NUMERIC'].dropna()
                if not valid_costs.empty:
                    analysis['total_spending'] = float(valid_costs.sum())
                    analysis['average_cost'] = float(valid_costs.mean())
                    analysis['median_cost'] = float(valid_costs.median())
                    analysis['min_cost'] = float(valid_costs.min())
                    analysis['max_cost'] = float(valid_costs.max())
            
            vendor_analysis[vendor] = analysis
        
        return vendor_analysis

    def _identify_patterns(self, search_results: List[Dict]) -> Dict[str, Any]:
        """
        Identify patterns in search results for recommendations
        """
        if not search_results:
            return {}
        
        df = pd.DataFrame(search_results)
        patterns = {}
        
        # Spending patterns
        if COST_COL in df.columns:
            df['COST_NUMERIC'] = pd.to_numeric(
                df[COST_COL].astype(str).str.replace(',', '').str.replace('$', ''),
                errors='coerce'
            )
            
            valid_costs = df['COST_NUMERIC'].dropna()
            if not valid_costs.empty:
                patterns['spending'] = {
                    'high_value_threshold': float(valid_costs.quantile(0.9)),
                    'low_value_threshold': float(valid_costs.quantile(0.1)),
                    'concentration': float(valid_costs.std() / valid_costs.mean()) if valid_costs.mean() > 0 else 0
                }
        
        # Vendor concentration
        if VENDOR_COL in df.columns:
            vendor_counts = df[VENDOR_COL].value_counts()
            total_records = len(df)
            
            patterns['vendor_concentration'] = {
                'top_vendor_share': float(vendor_counts.iloc[0] / total_records) if not vendor_counts.empty else 0,
                'top_5_share': float(vendor_counts.head(5).sum() / total_records) if len(vendor_counts) >= 5 else 0,
                'unique_vendors': len(vendor_counts)
            }
        
        # Category patterns
        if COMMODITY_COL in df.columns:
            category_counts = df[COMMODITY_COL].value_counts()
            patterns['categories'] = {
                'dominant_category': category_counts.index[0] if not category_counts.empty else None,
                'category_diversity': len(category_counts),
                'top_categories': category_counts.head(3).to_dict()
            }
        
        return patterns

    def _calculate_statistics(self, search_results: List[Dict]) -> Dict[str, Any]:
        """
        Calculate comprehensive statistics from search results
        """
        if not search_results:
            return {}
        
        df = pd.DataFrame(search_results)
        stats = {
            'total_records': len(df),
            'data_completeness': {}
        }
        
        # Check data completeness
        for col in df.columns:
            stats['data_completeness'][col] = float(df[col].notna().sum() / len(df))
        
        # Cost statistics
        if COST_COL in df.columns:
            df['COST_NUMERIC'] = pd.to_numeric(
                df[COST_COL].astype(str).str.replace(',', '').str.replace('$', ''),
                errors='coerce'
            )
            
            valid_costs = df['COST_NUMERIC'].dropna()
            if not valid_costs.empty:
                stats['cost_statistics'] = {
                    'count': len(valid_costs),
                    'total': float(valid_costs.sum()),
                    'mean': float(valid_costs.mean()),
                    'median': float(valid_costs.median()),
                    'std': float(valid_costs.std()),
                    'min': float(valid_costs.min()),
                    'max': float(valid_costs.max()),
                    'q25': float(valid_costs.quantile(0.25)),
                    'q75': float(valid_costs.quantile(0.75)),
                    'skewness': float(valid_costs.skew()),
                    'kurtosis': float(valid_costs.kurtosis())
                }
        
        # Vendor statistics
        if VENDOR_COL in df.columns:
            vendor_counts = df[VENDOR_COL].value_counts()
            stats['vendor_statistics'] = {
                'unique_vendors': len(vendor_counts),
                'records_per_vendor_mean': float(vendor_counts.mean()),
                'records_per_vendor_median': float(vendor_counts.median()),
                'most_frequent_vendor': vendor_counts.index[0] if not vendor_counts.empty else None,
                'most_frequent_count': int(vendor_counts.iloc[0]) if not vendor_counts.empty else 0
            }
        
        # Category statistics
        if COMMODITY_COL in df.columns:
            category_counts = df[COMMODITY_COL].value_counts()
            stats['category_statistics'] = {
                'unique_categories': len(category_counts),
                'most_common_category': category_counts.index[0] if not category_counts.empty else None,
                'category_distribution': category_counts.head(10).to_dict()
            }
        
        return stats

    def _calculate_confidence(self, search_results: List[Dict], response: str) -> int:
        """
        Calculate confidence score for the response
        """
        confidence = 50  # Base confidence
        
        # Adjust based on number of results
        if len(search_results) > 20:
            confidence += 20
        elif len(search_results) > 10:
            confidence += 10
        elif len(search_results) < 5:
            confidence -= 20
        
        # Adjust based on relevance scores
        if search_results:
            avg_relevance = np.mean([r.get('relevance_score', 0.5) for r in search_results[:10]])
            confidence += int(avg_relevance * 20)
        
        # Adjust based on response quality
        if response:
            if '$' in response and any(char.isdigit() for char in response):
                confidence += 10  # Contains specific numbers
            if len(response) > 500:
                confidence += 10  # Comprehensive response
            if 'unable' in response.lower() or 'cannot' in response.lower():
                confidence -= 15  # Indicates limitations
        
        # Cap between 0 and 100
        return max(0, min(100, confidence))

    def _calculate_relevance_score(self, search_results: List[Dict]) -> float:
        """
        Calculate overall relevance score for search results
        """
        if not search_results:
            return 0.0
        
        relevance_scores = [r.get('relevance_score', 0.5) for r in search_results[:10]]
        return float(np.mean(relevance_scores)) if relevance_scores else 0.0

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
    
    Args:
        question: User's query
        mode: Processing mode
        
    Returns:
        Dict containing answer and metadata
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
    """
    question = f"Provide a comprehensive analysis of {vendor_name} including spending patterns, categories, and relationships"
    return answer_question_intelligent(question, mode="analytical")

def get_recommendations(context: str = "cost optimization") -> Dict[str, Any]:
    """
    Get strategic recommendations based on semantic analysis
    """
    question = f"What are the top recommendations for {context} based on procurement patterns?"
    return answer_question_intelligent(question, mode="semantic")

def compare_vendors_semantic(vendors: List[str]) -> Dict[str, Any]:
    """
    Perform semantic comparison of multiple vendors
    """
    vendor_str = " and ".join(vendors)
    question = f"Compare {vendor_str} across all dimensions including spending, categories, and performance"
    return answer_question_intelligent(question, mode="semantic")

# ============================================
# TESTING
# ============================================

if __name__ == "__main__":
    # Test the enhanced RAG processor
    test_queries = [
        "What are the spending patterns for Dell?",
        "Compare Microsoft and IBM procurement strategies",
        "Which vendors should we optimize for cost reduction?",
        "Analyze technology procurement trends",
        "What's the relationship between order size and vendor performance?",
    ]
    
    print("Testing Enhanced RAG Processor")
    print("=" * 60)
    
    processor = get_rag_processor()
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 40)
        
        result = processor.process_query(query)
        
        print(f"Confidence: {result.get('confidence')}%")
        print(f"Records Analyzed: {result.get('records_analyzed')}")
        print(f"Search Relevance: {result.get('search_relevance', 0):.2f}")
        print(f"\nAnswer Preview:")
        print(result.get('answer', 'No answer')[:500] + "...")
        print("-" * 40)
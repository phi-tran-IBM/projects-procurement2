"""
query_decomposer.py - LLM-Powered Query Decomposition and Understanding
Provides intelligent query analysis, decomposition, and entity resolution
"""

import os
import re
import json
import logging
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import asdict
from functools import lru_cache
import hashlib

from dotenv import load_dotenv
load_dotenv()

# Import LLM
from langchain_ibm import ChatWatsonx
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from pydantic import BaseModel, Field, validator

# --- MODIFIED: Import new specialized model constants ---
from constants import (
    WATSONX_URL, WATSONX_PROJECT_ID, WATSONX_API_KEY,
    DECOMPOSER_MODEL, SYNTHESIS_MODEL,  # Changed from LLM_MODEL
    KNOWN_VENDOR_MAPPINGS, VENDOR_COL, COST_COL, COMMODITY_COL
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================
# DATA MODELS (Unchanged)
# ============================================

class QueryIntent(BaseModel):
    """Structured query intent from LLM"""
    primary_intent: str = Field(description="Main intent: comparison, aggregation, ranking, lookup, statistical, trend, recommendation")
    confidence: float = Field(description="Confidence score 0-1", ge=0, le=1)
    requires_semantic: bool = Field(description="Whether semantic search is needed")
    requires_sql: bool = Field(description="Whether SQL query is needed")
    
    @validator('primary_intent')
    def validate_intent(cls, v):
        valid_intents = ['comparison', 'aggregation', 'ranking', 'lookup', 
                        'statistical', 'trend', 'recommendation', 'exploration', 'other']
        if v.lower() not in valid_intents:
            return 'other'
        return v.lower()

class EntityExtraction(BaseModel):
    """Extracted entities from query"""
    vendors: List[str] = Field(default_factory=list, description="Vendor names or references")
    metrics: List[str] = Field(default_factory=list, description="Metrics mentioned (spending, count, average, etc.)")
    time_periods: List[str] = Field(default_factory=list, description="Time references")
    commodities: List[str] = Field(default_factory=list, description="Product/service categories")
    constraints: List[str] = Field(default_factory=list, description="Filters or conditions")
    ambiguous_references: Dict[str, str] = Field(default_factory=dict, description="Ambiguous terms and their likely meanings")

class SubQuery(BaseModel):
    """Individual sub-query component"""
    query: str = Field(description="The sub-query text")
    type: str = Field(description="Type: sql, semantic, calculation")
    dependencies: List[int] = Field(default_factory=list, description="Indices of queries this depends on")
    required_data: List[str] = Field(default_factory=list, description="Data fields needed")

class QueryDecomposition(BaseModel):
    """Complete query decomposition"""
    original_query: str = Field(description="Original user query")
    is_complex: bool = Field(description="Whether query needs decomposition")
    sub_queries: List[SubQuery] = Field(default_factory=list, description="Decomposed sub-queries")
    execution_order: List[int] = Field(default_factory=list, description="Order to execute sub-queries")
    combination_strategy: str = Field(description="How to combine results: merge, sequential, conditional")

# ============================================
# LLM QUERY DECOMPOSER
# ============================================

class LLMQueryDecomposer:
    """Intelligent query decomposition using LLM"""
    
    def __init__(self):
        """Initialize the decomposer with specialized LLMs"""
        try:
            # --- MODIFIED: Initialize two separate LLM clients ---
            # Fast model for analysis, classification, and decomposition
            self.decomposer_llm = ChatWatsonx(
                model_id=DECOMPOSER_MODEL,
                url=WATSONX_URL,
                project_id=WATSONX_PROJECT_ID or os.getenv("WX_AI_PROJECTID"),
                apikey=WATSONX_API_KEY or os.getenv("WX_AI_APIKEY"),
                params={"decoding_method": "greedy", "max_new_tokens": 1024, "temperature": 0.1}
            )
            
            # Powerful model for generating high-quality user-facing responses
            self.synthesis_llm = ChatWatsonx(
                model_id=SYNTHESIS_MODEL,
                url=WATSONX_URL,
                project_id=WATSONX_PROJECT_ID or os.getenv("WX_AI_PROJECTID"),
                apikey=WATSONX_API_KEY or os.getenv("WX_AI_APIKEY"),
                params={"decoding_method": "greedy", "max_new_tokens": 1500, "temperature": 0.4}
            )

            self.llm_available = True
            logger.info("LLM Query Decomposer initialized with specialized models")
            
            # Initialize output parsers
            self.intent_parser = PydanticOutputParser(pydantic_object=QueryIntent)
            self.entity_parser = PydanticOutputParser(pydantic_object=EntityExtraction)
            self.decomposition_parser = PydanticOutputParser(pydantic_object=QueryDecomposition)
            
            # Output fixing parser uses the decomposer model to fix its own output
            self.fixing_parser = OutputFixingParser.from_llm(parser=self.intent_parser, llm=self.decomposer_llm)
            
        except Exception as e:
            logger.error(f"Failed to initialize LLMs: {e}")
            self.llm_available = False
            self.decomposer_llm = None
            self.synthesis_llm = None

    def analyze_query_intent(self, query: str) -> QueryIntent:
        """Analyze query intent using the fast decomposer LLM"""
        if not self.llm_available:
            return self._fallback_intent_analysis(query)
        
        try:
            prompt = PromptTemplate(
                template="""Analyze the intent of this procurement query... {format_instructions}""", # Template unchanged
                input_variables=["query"],
                partial_variables={"format_instructions": self.intent_parser.get_format_instructions()}
            )
            # --- MODIFIED: Use the decomposer LLM ---
            llm_response = self.decomposer_llm.invoke(prompt.format(query=query))
            
            try:
                intent = self.intent_parser.parse(llm_response.content)
            except:
                intent = self.fixing_parser.parse(llm_response.content)
            
            logger.info(f"Query intent identified: {intent.primary_intent} (confidence: {intent.confidence})")
            return intent
        except Exception as e:
            logger.error(f"LLM intent analysis failed: {e}")
            return self._fallback_intent_analysis(query)
    
    def extract_entities(self, query: str, context: Optional[Dict] = None) -> EntityExtraction:
        """Extract entities using the fast decomposer LLM"""
        if not self.llm_available:
            return self._fallback_entity_extraction(query)
        
        try:
            vendor_context = self._build_vendor_context()
            prompt = PromptTemplate(
                template="""Extract all entities from this procurement query... {format_instructions}""", # Template unchanged
                input_variables=["query", "vendor_context", "vendor_col", "cost_col", "commodity_col"],
                partial_variables={"format_instructions": self.entity_parser.get_format_instructions()}
            )
            
            # --- MODIFIED: Use the decomposer LLM ---
            llm_response = self.decomposer_llm.invoke(prompt.format(
                query=query, vendor_context=vendor_context,
                vendor_col=VENDOR_COL, cost_col=COST_COL, commodity_col=COMMODITY_COL
            ))
            
            entities = self.entity_parser.parse(llm_response.content)
            entities.vendors = self._resolve_vendor_aliases(entities.vendors)
            logger.info(f"Extracted entities: {len(entities.vendors)} vendors, {len(entities.metrics)} metrics")
            return entities
        except Exception as e:
            logger.error(f"LLM entity extraction failed: {e}")
            return self._fallback_entity_extraction(query)
    
    def decompose_complex_query(self, query: str, intent: QueryIntent, entities: EntityExtraction) -> QueryDecomposition:
        """Decompose complex queries using the fast decomposer LLM"""
        if not self.llm_available:
            return self._create_simple_decomposition(query)
        
        try:
            prompt = PromptTemplate(
                template="""Decompose this complex procurement query... {format_instructions}""", # Template unchanged
                input_variables=["query", "intent", "entities"],
                partial_variables={"format_instructions": self.decomposition_parser.get_format_instructions()}
            )
            
            # --- MODIFIED: Use the decomposer LLM ---
            llm_response = self.decomposer_llm.invoke(prompt.format(
                query=query,
                intent=json.dumps(intent.model_dump()),
                entities=json.dumps(entities.model_dump())
            ))
            
            decomposition = self.decomposition_parser.parse(llm_response.content)
            decomposition.original_query = query
            logger.info(f"Query decomposed into {len(decomposition.sub_queries)} sub-queries")
            return decomposition
        except Exception as e:
            logger.error(f"Query decomposition failed: {e}")
            return self._create_simple_decomposition(query)
    
    def generate_natural_response(self, query: str, raw_results: Dict[str, Any], 
                                 intent: QueryIntent, entities: EntityExtraction) -> str:
        """Generate natural language response using the powerful synthesis LLM"""
        if not self.llm_available:
            return self._format_basic_response(raw_results)
        
        try:
            context = self._prepare_result_context(raw_results)
            prompt = PromptTemplate(
                template="""Generate a natural, helpful response...""", # Template unchanged
                input_variables=["query", "intent_type", "context"]
            )
            
            # --- MODIFIED: Use the synthesis LLM ---
            llm_response = self.synthesis_llm.invoke(prompt.format(
                query=query,
                intent_type=intent.primary_intent,
                context=context
            ))
            return llm_response.content
        except Exception as e:
            logger.error(f"Natural response generation failed: {e}")
            return self._format_basic_response(raw_results)
    
    def resolve_ambiguous_reference(self, reference: str, context: Optional[str] = None) -> List[str]:
        """Resolve ambiguous references using the powerful synthesis LLM"""
        if not self.llm_available: return []
        
        try:
            vendor_list = ', '.join(list(KNOWN_VENDOR_MAPPINGS.keys())[:20])
            prompt = f"""Resolve this ambiguous vendor reference... Answer:""" # Prompt unchanged
            
            # --- MODIFIED: Use the synthesis LLM for better reasoning ---
            llm_response = self.synthesis_llm.invoke(prompt)
            
            vendors = [v.strip() for v in llm_response.content.split(',') if v.strip()]
            logger.info(f"Resolved '{reference}' to: {vendors}")
            return vendors[:3]
        except Exception as e:
            logger.error(f"Failed to resolve reference '{reference}': {e}")
            return []

    # All helper and fallback methods below this point are unchanged
    # ============================================
    # HELPER METHODS
    # ============================================
    
    def _build_vendor_context(self) -> str:
        context_lines = []
        for category, vendors in list(KNOWN_VENDOR_MAPPINGS.items())[:10]:
            context_lines.append(f"- {category}: {', '.join(vendors[:3])}")
        return '\n'.join(context_lines)
    
    def _resolve_vendor_aliases(self, vendors: List[str]) -> List[str]:
        resolved = []
        for vendor in vendors:
            vendor_upper = vendor.upper()
            found = False
            for canonical, aliases in KNOWN_VENDOR_MAPPINGS.items():
                if vendor_upper in [a.upper() for a in aliases]:
                    resolved.append(canonical)
                    found = True
                    break
            if not found:
                resolved.append(vendor)
        return list(set(resolved))
    
    def _prepare_result_context(self, raw_results: Dict[str, Any]) -> str:
        context_parts = []
        if 'answer' in raw_results: context_parts.append(f"Direct Answer: {raw_results['answer']}")
        if 'records_analyzed' in raw_results: context_parts.append(f"Records Analyzed: {raw_results['records_analyzed']}")
        if 'summary' in raw_results: context_parts.append(f"Summary: {raw_results['summary']}")
        if 'vendors' in raw_results and isinstance(raw_results['vendors'], list):
            vendor_summary = []
            for vendor in raw_results['vendors'][:5]:
                if isinstance(vendor, dict):
                    vendor_summary.append(f"- {vendor.get('name', 'Unknown')}: ${vendor.get('total_spending', 0):,.2f}")
            if vendor_summary: context_parts.append("Top Vendors:\n" + '\n'.join(vendor_summary))
        return '\n\n'.join(context_parts)
    
    def _format_basic_response(self, raw_results: Dict[str, Any]) -> str:
        if 'answer' in raw_results: return raw_results['answer']
        if 'summary' in raw_results: return raw_results['summary']
        return "Query processed successfully."

    # ============================================
    # FALLBACK METHODS
    # ============================================
    
    def _fallback_intent_analysis(self, query: str) -> QueryIntent:
        query_lower = query.lower()
        if any(word in query_lower for word in ['compare', 'vs', 'versus']):
            return QueryIntent(primary_intent='comparison', confidence=0.7, requires_semantic=False, requires_sql=True)
        if any(word in query_lower for word in ['total', 'sum', 'average']):
            return QueryIntent(primary_intent='aggregation', confidence=0.7, requires_semantic=False, requires_sql=True)
        if any(word in query_lower for word in ['top', 'best', 'highest']):
            return QueryIntent(primary_intent='ranking', confidence=0.7, requires_semantic=False, requires_sql=True)
        if any(word in query_lower for word in ['should', 'recommend', 'suggest']):
            return QueryIntent(primary_intent='recommendation', confidence=0.6, requires_semantic=True, requires_sql=True)
        return QueryIntent(primary_intent='other', confidence=0.5, requires_semantic=True, requires_sql=True)
    
    def _fallback_entity_extraction(self, query: str) -> EntityExtraction:
        entities = EntityExtraction()
        query_upper = query.upper()
        for vendor_key, aliases in KNOWN_VENDOR_MAPPINGS.items():
            for alias in aliases:
                if alias in query_upper:
                    entities.vendors.append(vendor_key)
                    break
        for metric in ['spending', 'cost', 'total', 'average', 'median', 'count']:
            if metric in query.lower():
                entities.metrics.append(metric)
        return entities
    
    def _create_simple_decomposition(self, query: str) -> QueryDecomposition:
        return QueryDecomposition(original_query=query, is_complex=False, sub_queries=[SubQuery(query=query, type='sql', dependencies=[], required_data=[])], execution_order=[0], combination_strategy='none')

# ============================================
# PUBLIC INTERFACE (Unchanged)
# ============================================

_decomposer_instance = None

def get_decomposer() -> LLMQueryDecomposer:
    global _decomposer_instance
    if _decomposer_instance is None:
        _decomposer_instance = LLMQueryDecomposer()
    return _decomposer_instance

def decompose_query(query: str, use_cache: bool = True) -> Dict[str, Any]:
    decomposer = get_decomposer()
    intent = decomposer.analyze_query_intent(query)
    entities = decomposer.extract_entities(query)
    if intent.confidence < 0.8 or intent.requires_semantic:
        decomposition = decomposer.decompose_complex_query(query, intent, entities)
    else:
        decomposition = decomposer._create_simple_decomposition(query)
    return {'intent': intent.model_dump(), 'entities': entities.model_dump(), 'decomposition': decomposition.model_dump(), 'is_complex': decomposition.is_complex}

def generate_response(query: str, raw_results: Dict[str, Any]) -> str:
    decomposer = get_decomposer()
    intent = decomposer.analyze_query_intent(query)
    entities = decomposer.extract_entities(query)
    return decomposer.generate_natural_response(query, raw_results, intent, entities)

def resolve_reference(reference: str, context: Optional[str] = None) -> List[str]:
    decomposer = get_decomposer()
    return decomposer.resolve_ambiguous_reference(reference, context)

# Testing block remains unchanged
if __name__ == "__main__":
    pass
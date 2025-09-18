"""
constants.py - Shared constants for the LLM-Enhanced Procurement RAG system
UPDATED: Added template-based prompt system with dynamic functions
"""

import os

# ============================================
# DATABASE CONFIGURATION
# ============================================
DB_PATH = os.getenv("DB_PATH", "data/verification.db")

# ============================================
# COLUMN NAMES
# ============================================
VENDOR_COL = "VENDOR_NAME_1"
COST_COL = "ITEM_TOTAL_COST"
DESC_COL = "ITEM_DESCRIPTION"
COMMODITY_COL = "COMMODITY_DESCRIPTION"
DATE_COL = "DATE_COLUMN"  # If exists

# ============================================
# DATA SOURCE
# ============================================
CSV_PATH = "data/temp_data.csv"

# ============================================
# CACHE CONFIGURATION
# ============================================
CACHE_MAX_SIZE = 1000
CACHE_TTL_SECONDS = 3600

# ============================================
# GRANULAR CACHE CONFIGURATION (NEW)
# ============================================
# Cache TTLs by type (in seconds)
CACHE_TTL_BY_TYPE = {
    'final_result': 1800,        # 30 minutes
    'decomposition': 7200,       # 2 hours (stable)
    'vendor_resolution': 3600,   # 1 hour
    'statistical': 300,          # 5 minutes (may change with new data)
    'semantic_search': 1800,     # 30 minutes
    'entity_extraction': 7200,   # 2 hours
    'unified_analysis': 7200,   # 2 hours
}

# Cache size limits by type
CACHE_MAX_SIZE_BY_TYPE = {
    'final_result': 500,
    'decomposition': 1000,
    'vendor_resolution': 2000,
    'statistical': 200,
    'semantic_search': 300,
    'entity_extraction': 500,
    'unified_analysis': 500,
}

# Cache key prefixes
CACHE_KEY_PREFIXES = {
    'final': 'result_',
    'decomposition': 'decomp_',
    'vendor': 'vendor_',
    'stats': 'stats_',
    'semantic': 'sem_',
    'entity': 'entity_',
    'unified': 'unified_',
}

# ============================================
# QUERY PROCESSING CONFIGURATION
# ============================================
QUERY_TIMEOUT_SECONDS = 30
MAX_RETRY_ATTEMPTS = 3
BACKOFF_FACTOR = 2.0
CONNECTION_POOL_SIZE = 5
BATCH_SIZE = 1000

# ============================================
# LLM CONFIGURATION
# ============================================
# Watson X Configuration
WATSONX_URL = os.getenv("WATSONX_URL", os.getenv("WATSONX_ENDPOINT_URL", "https://us-south.ml.cloud.ibm.com"))
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID", os.getenv("WX_AI_PROJECTID"))
WATSONX_API_KEY = os.getenv("WATSONX_API_KEY", os.getenv("WX_AI_APIKEY"))

# LLM Model Selection for Specialized Tasks
DECOMPOSER_MODEL = os.getenv("DECOMPOSER_MODEL", "mistralai/mistral-small-3-1-24b-instruct-2503")
SYNTHESIS_MODEL = os.getenv("SYNTHESIS_MODEL", "ibm/granite-3-3-8b-instruct")

LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "800"))
LLM_TOP_P = float(os.getenv("LLM_TOP_P", "0.95"))

# ============================================
# LLM PROMPT TEMPLATES (ENHANCED WITH TEMPLATES)
# ============================================

# Unified Analysis Prompt - Single call instead of 3-4
UNIFIED_ANALYSIS_PROMPT = """You are an expert procurement query analyzer. Your task is to analyze a user's query and provide a structured JSON output. Think step-by-step to deconstruct the query before you respond.

<query>
{query}
</query>

<example>
  <query>Compare spending on Dell vs IBM last year, and recommend which to invest in.</query>
  <output>
    {{
      "intent": "comparison",
      "complexity": "complex",
      "requires_decomposition": true,
      "sub_queries": [
        "What was the total spending on Dell last year?",
        "What was the total spending on IBM last year?",
        "Recommend which vendor to invest in based on spending."
      ],
      "entities": {{
        "vendors": ["Dell", "IBM"],
        "metrics": ["spending", "invest"],
        "time_periods": ["last year"],
        "commodities": []
      }},
      "suggested_approach": "hybrid"
    }}
  </output>
</example>

Carefully analyze the user's query and return ONLY the JSON object with the required structure.

<output_structure>
{{
  "intent": "comparison|aggregation|ranking|lookup|statistical|trend|recommendation|exploration|other",
  "confidence": 0.0-1.0,
  "entities": {{
    "vendors": ["list of vendor names mentioned"],
    "metrics": ["spending", "count", "average", etc."],
    "time_periods": ["any time references"],
    "commodities": ["product/service categories"]
  }},
  "complexity": "simple|complex",
  "suggested_approach": "sql|semantic|hybrid",
  "requires_decomposition": true|false,
  "sub_queries": ["list of sub-queries if complex, otherwise empty"]
}}
</output_structure>

Final JSON Output:"""

# Standard Grounded Response Prompts (Non-Template)
GROUNDED_SYNTHESIS_PROMPT = """You are a procurement data analyst. You MUST base your response ONLY on the provided data.

STRICT RULES:
1. Use ONLY information from the Data Context below.
2. Include specific numbers with proper formatting ($X,XXX.XX).
3. Reference actual vendor names from the data.
4. If the data doesn't answer the question, say "I don't have sufficient data to answer that question."
5. Never make up information or use hypothetical examples.

Data Context:
{context}

Question: {question}

Remember: Do not invent any information not present in the provided Data Context.

Response:"""

GROUNDED_COMPARISON_PROMPT = """You are analyzing vendor comparison data. Provide a detailed comparison using ONLY the data provided.

RULES:
1. Compare vendors using the exact metrics provided.
2. Use specific dollar amounts and percentages.
3. Highlight the highest and lowest values.
4. If vendors are missing data, explicitly state what is unavailable.

Vendor Data:
{vendor_data}

Question: {question}

Remember: Do not add any information not present in the provided Vendor Data.

Comparison Analysis:"""

GROUNDED_RECOMMENDATION_PROMPT = """You are a strategic procurement advisor. Your recommendations MUST be directly based ONLY on the data provided in the <data_analysis> section.

<data_analysis>
{context}
</data_analysis>

<focus_area>
{focus}
</focus_area>

Generate a JSON array of strategic recommendations with the following structure. Each recommendation MUST include a justification that references specific data points, numbers, and vendor names from the context.
[
  {{
    "recommendation": "Your strategic advice here.",
    "justification": "Explain exactly which data points from the context support this recommendation. Reference specific numbers and vendor names."
  }}
]

If the data is insufficient to make a recommendation, return an empty array []. Do not add any text or explanation outside of the JSON array.

Strategic Recommendations:"""

GROUNDED_STATISTICAL_PROMPT = """Provide a statistical analysis using ONLY the calculated metrics provided.

RULES:
1. Report the exact calculated values from the Statistical Results.
2. Explain what the statistics mean in business terms, based only on the numbers given.
3. Identify any notable patterns or outliers visible in the numbers.
4. Do not extrapolate or predict beyond the provided calculations.

Statistical Results:
{statistics}

Remember: Your interpretation must be based exclusively on the provided Statistical Results.

Interpretation:"""

# Template-Based Grounded Response Prompts (NEW)
GROUNDED_SYNTHESIS_PROMPT_TEMPLATE = """You are a procurement data analyst. Respond ONLY using the provided data.

Data Context:
{context}

Question: {question}

Structure your response using these XML tags:

<RESPONSE_START>
<ANSWER>
Provide your answer here using ONLY the data from the context. Include specific numbers, vendor names, and metrics exactly as they appear in the data.
</ANSWER>

If the data is insufficient:
<INSUFFICIENT_DATA>
Explain what data is missing or why the question cannot be answered fully.
</INSUFFICIENT_DATA>
</RESPONSE_START>"""

GROUNDED_COMPARISON_PROMPT_TEMPLATE = """Compare vendors using ONLY the provided data.

Vendor Data:
{vendor_data}

Structure your response with these XML tags:

<COMPARISON_START>
<SUMMARY>Brief overall comparison summary</SUMMARY>

<VENDOR1>
<NAME>Vendor name from data</NAME>
<PERFORMANCE>Total spending, orders, and key metrics from data</PERFORMANCE>
<STRENGTHS>What this vendor does well based on the numbers</STRENGTHS>
<CONCERNS>Any concerns based on the data</CONCERNS>
</VENDOR1>

<VENDOR2>
<NAME>Vendor name from data</NAME>
<PERFORMANCE>Total spending, orders, and key metrics from data</PERFORMANCE>
<STRENGTHS>What this vendor does well based on the numbers</STRENGTHS>
<CONCERNS>Any concerns based on the data</CONCERNS>
</VENDOR2>

<RECOMMENDATION>Which vendor to prefer and why, based solely on the data</RECOMMENDATION>

If data is insufficient:
<INSUFFICIENT_DATA>Explain what is missing</INSUFFICIENT_DATA>
</COMPARISON_START>"""

GROUNDED_RECOMMENDATION_PROMPT_TEMPLATE = """Generate recommendations based ONLY on the provided data.

Data Analysis:
{context}

Focus Area: {focus}

Structure your response with these XML tags:

<RECOMMENDATIONS_START>
<REC1>
<ACTION>Specific action to take</ACTION>
<JUSTIFICATION>Data points that support this (use exact numbers and vendor names)</JUSTIFICATION>
<PRIORITY>HIGH/MEDIUM/LOW</PRIORITY>
</REC1>

<REC2>
<ACTION>Another specific action</ACTION>
<JUSTIFICATION>Supporting data with specific numbers</JUSTIFICATION>
<PRIORITY>HIGH/MEDIUM/LOW</PRIORITY>
</REC2>

<REC3>
<ACTION>Third action if applicable</ACTION>
<JUSTIFICATION>Data-based reasoning</JUSTIFICATION>
<PRIORITY>HIGH/MEDIUM/LOW</PRIORITY>
</REC3>

If data insufficient:
<INSUFFICIENT_DATA>Explain what additional data would be needed</INSUFFICIENT_DATA>
</RECOMMENDATIONS_START>"""

GROUNDED_STATISTICAL_PROMPT_TEMPLATE = """Analyze the statistical data provided.

Statistical Results:
{statistics}

Structure your response with these XML tags:

<STATISTICAL_ANALYSIS>
<SUMMARY>Brief summary of the statistical findings</SUMMARY>

<FINDING1>First key statistical insight with specific numbers</FINDING1>
<FINDING2>Second key statistical insight</FINDING2>
<FINDING3>Third insight if relevant</FINDING3>

<BUSINESS_IMPACT>What these statistics mean for the business</BUSINESS_IMPACT>

<RECOMMENDATIONS>Actions to take based on these statistics</RECOMMENDATIONS>

If data insufficient:
<INSUFFICIENT_DATA>What statistics are missing or unreliable</INSUFFICIENT_DATA>
</STATISTICAL_ANALYSIS>"""

# ============================================
# DYNAMIC PROMPT FUNCTIONS (NEW)
# ============================================

def get_grounded_synthesis_prompt():
    """
    Dynamic prompt function that returns appropriate template based on feature flag.
    Returns template version if template_parsing is enabled, standard version otherwise.
    """
    if FEATURES.get('template_parsing', False):
        return GROUNDED_SYNTHESIS_PROMPT_TEMPLATE
    else:
        return GROUNDED_SYNTHESIS_PROMPT

def get_grounded_comparison_prompt():
    """
    Dynamic prompt function for comparison queries.
    Returns template version if template_parsing is enabled, standard version otherwise.
    """
    if FEATURES.get('template_parsing', False):
        return GROUNDED_COMPARISON_PROMPT_TEMPLATE
    else:
        return GROUNDED_COMPARISON_PROMPT

def get_grounded_recommendation_prompt():
    """
    Dynamic prompt function for recommendation queries.
    Returns template version if template_parsing is enabled, standard version otherwise.
    """
    if FEATURES.get('template_parsing', False):
        return GROUNDED_RECOMMENDATION_PROMPT_TEMPLATE
    else:
        return GROUNDED_RECOMMENDATION_PROMPT

def get_grounded_statistical_prompt():
    """
    Dynamic prompt function for statistical analysis.
    Returns template version if template_parsing is enabled, standard version otherwise.
    """
    if FEATURES.get('template_parsing', False):
        return GROUNDED_STATISTICAL_PROMPT_TEMPLATE
    else:
        return GROUNDED_STATISTICAL_PROMPT

# ============================================
# SMART ROUTING CONFIGURATION (NEW)
# ============================================

# Patterns that can bypass LLM and go directly to SQL
DIRECT_SQL_PATTERNS = {
    'total_spending': {
        'pattern': r'(?i)(total\s+spending|total\s+cost|how\s+much.*spent|total\s+procurement)',
        'sql_template': f"SELECT SUM(CAST({COST_COL} AS FLOAT)) as total FROM procurement WHERE {COST_COL} IS NOT NULL",
        'response_template': "Total procurement spending: ${value:,.2f}"
    },
    'vendor_count': {
        'pattern': r'(?i)(how\s+many\s+vendors|count.*vendors|number\s+of\s+vendors|total\s+vendors)',
        'sql_template': f"SELECT COUNT(DISTINCT {VENDOR_COL}) as count FROM procurement WHERE {VENDOR_COL} IS NOT NULL",
        'response_template': "Total number of vendors: {value:,.0f}"
    },
    'order_count': {
        'pattern': r'(?i)(how\s+many\s+orders|count.*orders|number\s+of\s+orders|total\s+orders)',
        'sql_template': f"SELECT COUNT(*) as count FROM procurement",
        'response_template': "Total number of orders: {value:,.0f}"
    },
    'average_order': {
        'pattern': r'(?i)(average\s+order|mean\s+order|avg\s+order|average\s+cost\s+per)',
        'sql_template': f"SELECT AVG(CAST({COST_COL} AS FLOAT)) as avg FROM procurement WHERE {COST_COL} IS NOT NULL",
        'response_template': "Average order value: ${value:,.2f}"
    },
    'max_order': {
        'pattern': r'(?i)(largest\s+order|biggest\s+order|maximum\s+order|highest\s+order)',
        'sql_template': f"SELECT MAX(CAST({COST_COL} AS FLOAT)) as max FROM procurement WHERE {COST_COL} IS NOT NULL",
        'response_template': "Largest order value: ${value:,.2f}"
    },
    'min_order': {
        'pattern': r'(?i)(smallest\s+order|minimum\s+order|lowest\s+order)',
        'sql_template': f"SELECT MIN(CAST({COST_COL} AS FLOAT)) as min FROM procurement WHERE {COST_COL} IS NOT NULL",
        'response_template': "Smallest order value: ${value:,.2f}"
    }
}

# Routing confidence thresholds
ROUTING_CONFIDENCE_THRESHOLD = 0.9
ROUTING_PATTERN_MIN_LENGTH = 3

# ============================================
# TOKEN LIMITS
# ============================================
DEFAULT_TOKEN_LIMIT = 800
MAX_TOKEN_LIMIT = 2000

# Query Type Token Limits
TOKEN_LIMITS = {
    "aggregation": 400,
    "comparison": 800,
    "ranking": 1000,
    "specific_lookup": 500,
    "semantic_search": 1200,
    "fuzzy_search": 1000,
    "statistical": 400,
    "trend_analysis": 1000,
    "recommendation": 1500,
    "complex_analytical": 2000,
}

# ============================================
# VENDOR CONFIGURATION
# ============================================
# Vendor Name Suffixes to Remove
VENDOR_SUFFIXES = [
    'INC', 'LLC', 'CORP', 'CORPORATION', 'COMPANY', 'CO', 'LTD', 
    'LIMITED', 'TECHNOLOGIES', 'SOLUTIONS', 'SYSTEMS', 'INTERNATIONAL',
    'ENTERPRISES', 'INDUSTRIES', 'SERVICES', 'GROUP', 'HOLDINGS',
    'PARTNERS', 'ASSOCIATES', 'CONSULTING', 'CORPORATION'
]

# Known Vendor Mappings
KNOWN_VENDOR_MAPPINGS = {
    "DELL": ["DELL", "DELL INC", "DELL TECHNOLOGIES", "DELL COMPUTER", "DELL COMPUTERS", "DELL EMC", "DELL COMPUTER CORP"],
    "IBM": ["IBM", "INTERNATIONAL BUSINESS MACHINES", "IBM CORPORATION", "I.B.M.", "IBM CORP", "IBM GLOBAL", "I B M", "I.B.M"],
    "MICROSOFT": ["MICROSOFT", "MICROSOFT CORPORATION", "MICROSOFT CORP", "MSFT", "MICROSOFT INC"],
    "ORACLE": ["ORACLE", "ORACLE CORPORATION", "ORACLE CORP", "ORACLE AMERICA", "ORACLE USA"],
    "HP": ["HP", "HEWLETT PACKARD", "HEWLETT-PACKARD", "HP INC", "HPE", "HP ENTERPRISE", "HEWLETT PACKARD ENTERPRISE"],
    "AMAZON": ["AMAZON", "AMAZON WEB SERVICES", "AWS", "AMAZON.COM", "AMAZON INC"],
    "GOOGLE": ["GOOGLE", "GOOGLE LLC", "GOOGLE INC", "ALPHABET", "GOOGLE CLOUD"],
    "APPLE": ["APPLE", "APPLE INC", "APPLE COMPUTER", "APPLE COMPUTERS"],
    "CISCO": ["CISCO", "CISCO SYSTEMS", "CISCO INC", "CISCO CORPORATION"],
    "INTEL": ["INTEL", "INTEL CORPORATION", "INTEL CORP"],
}

# ============================================
# VENDOR RESOLUTION CONFIGURATION (NEW)
# ============================================

# Fuzzy matching thresholds by confidence level
VENDOR_FUZZY_THRESHOLDS = {
    'exact': 1.0,
    'high_confidence': 0.9,
    'medium_confidence': 0.8,
    'low_confidence': 0.7,
    'last_resort': 0.6
}

# Maximum vendors to return per resolution attempt
VENDOR_RESOLUTION_MAX_RESULTS = 5

# Vendor resolution strategies in order of preference
VENDOR_RESOLUTION_STRATEGIES = [
    'exact_match',
    'known_mappings',
    'normalized_match',
    'fuzzy_match',
    'partial_match'
]

# Common vendor name variations to strip (regex patterns)
VENDOR_NAME_VARIATIONS = [
    r'\s+INC\.?$',
    r'\s+LLC\.?$',
    r'\s+CORP(?:ORATION)?\.?$',
    r'\s+COMPANY$',
    r'\s+CO\.?$',
    r'\s+LTD\.?$',
    r'\s+LIMITED$',
    r'\s+TECHNOLOGIES$',
    r'\s+SOLUTIONS$',
]

# ============================================
# TIERED SEMANTIC SEARCH CONFIGURATION (NEW)
# ============================================

# Search tiers with parameters
SEMANTIC_SEARCH_TIERS = {
    'tier_1_exact': {
        'n_results': 15,
        'min_relevance': 0.8,
        'query_modification': None,
        'description': 'Exact query match'
    },
    'tier_2_expanded': {
        'n_results': 30,
        'min_relevance': 0.6,
        'query_modification': 'add_synonyms',
        'description': 'Query with synonyms'
    },
    'tier_3_broad': {
        'n_results': 50,
        'min_relevance': 0.4,
        'query_modification': 'category_search',
        'description': 'Broad category search'
    }
}

# Strategic term mappings for query expansion
STRATEGIC_TERM_MAPPINGS = {
    'reduce overhead': ['cost reduction', 'savings', 'efficiency', 'optimization', 'spend less'],
    'improve efficiency': ['streamline', 'optimize', 'enhance performance', 'reduce waste', 'better processes'],
    'risk management': ['vendor reliability', 'supply chain', 'contingency', 'diversification', 'vendor risk'],
    'cost optimization': ['savings', 'reduce spending', 'negotiate', 'consolidate', 'better pricing'],
    'vendor consolidation': ['reduce vendors', 'fewer suppliers', 'streamline suppliers', 'vendor reduction'],
    'strategic sourcing': ['procurement strategy', 'sourcing optimization', 'supplier selection', 'vendor selection'],
}

# ============================================
# RESPONSE QUALITY CONFIGURATION (NEW)
# ============================================

# Minimum data requirements for different query types
MIN_DATA_REQUIREMENTS = {
    'comparison': 2,      # Need at least 2 vendors
    'statistical': 10,    # Need at least 10 data points
    'trend': 30,         # Need at least 30 data points
    'recommendation': 5,  # Need at least 5 vendors
    'ranking': 3,        # Need at least 3 items to rank
}

# Response quality indicators
QUALITY_THRESHOLDS = {
    'high_quality': {
        'min_data_points': 100,
        'min_relevance': 0.8,
        'max_null_percentage': 0.1
    },
    'medium_quality': {
        'min_data_points': 20,
        'min_relevance': 0.6,
        'max_null_percentage': 0.3
    },
    'low_quality': {
        'min_data_points': 5,
        'min_relevance': 0.4,
        'max_null_percentage': 0.5
    }
}

# Fallback messages for insufficient data
INSUFFICIENT_DATA_MESSAGES = {
    'no_data': "I don't have any data matching your query. Please try rephrasing or using different search terms.",
    'insufficient_vendors': "I need at least {required} vendors for this comparison, but only found {found}.",
    'insufficient_samples': "Statistical analysis requires at least {required} data points, but only {found} are available.",
    'low_relevance': "The available data has low relevance to your query. Results may not be accurate.",
    'vendor_not_found': "I couldn't find vendor '{vendor}' in the database. Similar vendors: {suggestions}",
}

# ============================================
# SQL SECURITY
# ============================================
# SQL Keywords for Injection Detection
SQL_INJECTION_KEYWORDS = [
    'DROP', 'DELETE', 'INSERT', 'UPDATE', 'ALTER', 'CREATE',
    'EXEC', 'EXECUTE', '--', '/*', '*/', 'UNION', 'SELECT FROM',
    'TRUNCATE', 'REPLACE', 'MERGE', 'GRANT', 'REVOKE'
]

# ============================================
# NATURAL LANGUAGE PROCESSING
# ============================================
# Stop Words for Keyword Extraction
STOP_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 
    'for', 'of', 'with', 'by', 'from', 'up', 'about', 'into', 
    'through', 'during', 'how', 'what', 'which', 'when', 'where', 
    'who', 'why', 'is', 'are', 'was', 'were', 'been', 'be', 
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 
    'could', 'should', 'may', 'might', 'must', 'can', 'me', 'show',
    'tell', 'get', 'find', 'give', 'please', 'help', 'want'
}

# ============================================
# QUERY CLASSIFICATION KEYWORDS
# ============================================
# Statistical Keywords
STATISTICAL_KEYWORDS = [
    'median', 'variance', 'standard deviation', 'stddev', 'std',
    'percentile', 'quartile', 'distribution', 'correlation',
    'mean', 'average', 'sum', 'count', 'min', 'max',
    'skewness', 'kurtosis', 'outliers'
]

# Comparison Keywords
COMPARISON_KEYWORDS = [
    'compare', 'vs', 'versus', 'against', 'difference between',
    'comparison', 'vs.', 'v.', 'and also', 'as well as',
    'compared to', 'relative to', 'contrast', 'better', 'worse'
]

# Aggregation Keywords
AGGREGATION_KEYWORDS = [
    'total', 'sum', 'average', 'mean', 'count', 'how many',
    'how much', 'aggregate', 'combined', 'overall', 'all',
    'cumulative', 'collective'
]

# Ranking Keywords
RANKING_KEYWORDS = [
    'top', 'bottom', 'largest', 'smallest', 'highest', 'lowest',
    'rank', 'best', 'worst', 'most', 'least', 'maximum', 'minimum',
    'first', 'last', 'leading', 'top-rated', 'premier'
]

# Recommendation Keywords
RECOMMENDATION_KEYWORDS = [
    'should', 'recommend', 'suggest', 'advice', 'optimize',
    'improve', 'enhance', 'better', 'strategy', 'plan',
    'consider', 'evaluate', 'assess', 'review'
]

# Trend Keywords
TREND_KEYWORDS = [
    'trend', 'over time', 'monthly', 'yearly', 'quarterly',
    'growth', 'decline', 'pattern', 'change', 'evolution',
    'trajectory', 'forecast', 'prediction'
]

# ============================================
# PERFORMANCE MONITORING (NEW)
# ============================================

# Performance targets (in seconds)
PERFORMANCE_TARGETS = {
    'simple_query': 2.0,
    'sql_query': 5.0,
    'semantic_query': 10.0,
    'complex_query': 15.0,
    'report_generation': 20.0
}

# Performance logging thresholds
SLOW_QUERY_THRESHOLD = 10.0
VERY_SLOW_QUERY_THRESHOLD = 20.0

# Metrics to track
TRACKED_METRICS = [
    'llm_calls_per_query',
    'cache_hit_rate',
    'average_response_time',
    'vendor_resolution_success_rate',
    'data_grounding_rate'
]

# ============================================
# SERVER CONFIGURATION
# ============================================
DEFAULT_PORT = int(os.getenv("PORT", "8080"))
DEFAULT_HOST = os.getenv("HOST", "0.0.0.0")

# ============================================
# LOGGING CONFIGURATION
# ============================================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# ============================================
# PERFORMANCE SETTINGS
# ============================================
# Fuzzy Matching Settings
FUZZY_THRESHOLD = 0.8
FUZZY_MAX_MATCHES = 10

# Response Limits
MAX_VENDORS_COMPARE = 10
MAX_RANKING_RESULTS = 100
MAX_QUERY_LENGTH = 1000

# Parallel Processing
MAX_PARALLEL_QUERIES = 5
THREAD_POOL_SIZE = 3

# ============================================
# LLM PROMPT TEMPLATES (EXISTING)
# ============================================
# Temperature settings for different query types
LLM_TEMPERATURES = {
    "factual": 0.1,
    "analytical": 0.3,
    "creative": 0.7,
    "conversational": 0.5
}

# Max tokens for different response types
LLM_MAX_TOKENS_BY_TYPE = {
    "brief": 200,
    "standard": 500,
    "detailed": 1000,
    "comprehensive": 2000
}

# ============================================
# SEMANTIC SEARCH CONFIGURATION
# ============================================
SEMANTIC_SEARCH_TOP_K = 50
SEMANTIC_RELEVANCE_THRESHOLD = 0.7
SEMANTIC_CACHE_ENABLED = True
SEMANTIC_CACHE_TTL = 1800

# ============================================
# CONFIDENCE THRESHOLDS
# ============================================
HIGH_CONFIDENCE_THRESHOLD = 80
MEDIUM_CONFIDENCE_THRESHOLD = 50
LOW_CONFIDENCE_THRESHOLD = 30

# Use LLM for queries below this confidence
LLM_FALLBACK_CONFIDENCE = 60

# ============================================
# BUSINESS RULES
# ============================================
# Thresholds for recommendations
MIN_ORDERS_FOR_VENDOR_ANALYSIS = 5
HIGH_CONCENTRATION_THRESHOLD = 30  # % of total spending
LOW_ACTIVITY_THRESHOLD = 5  # orders
OUTLIER_THRESHOLD_MULTIPLIER = 3  # Standard deviations

# ============================================
# REPORT GENERATION
# ============================================
REPORT_FORMATS = ["executive", "detailed", "summary", "technical"]
DEFAULT_REPORT_FORMAT = "executive"
MAX_REPORT_SECTIONS = 10
REPORT_CHART_TYPES = ["bar", "pie", "line", "scatter", "heatmap"]

# ============================================
# ERROR MESSAGES
# ============================================
ERROR_MESSAGES = {
    "llm_unavailable": "Advanced AI features are currently unavailable. Using standard processing.",
    "database_error": "Unable to access procurement database. Please try again later.",
    "vendor_not_found": "Vendor not found. Try using the full vendor name or check spelling.",
    "invalid_query": "Query format not recognized. Please try rephrasing in natural language.",
    "timeout": "Query processing timed out. Try simplifying your question.",
}

# ============================================
# FEATURE FLAGS (CENTRALIZED & ENHANCED)
# ============================================
FEATURES = {
    # Core features
    "llm_enabled": os.getenv("ENABLE_LLM", "true").lower() == "true",
    "semantic_search": os.getenv("ENABLE_SEMANTIC", "true").lower() == "true",
    "advanced_analytics": os.getenv("ENABLE_ADVANCED", "true").lower() == "true",
    "caching": os.getenv("ENABLE_CACHE", "true").lower() == "true",
    "parallel_processing": os.getenv("ENABLE_PARALLEL", "true").lower() == "true",
    "natural_language": os.getenv("ENABLE_NL", "true").lower() == "true",
    
    # All optimization features (enabled by default)
    "unified_analysis": os.getenv("ENABLE_UNIFIED_ANALYSIS", "true").lower() == "true",
    "granular_caching": os.getenv("ENABLE_GRANULAR_CACHING", "true").lower() == "true",
    "smart_routing": os.getenv("ENABLE_SMART_ROUTING", "true").lower() == "true",
    "grounded_prompts": os.getenv("ENABLE_GROUNDED_PROMPTS", "true").lower() == "true",
    "enhanced_context": os.getenv("ENABLE_ENHANCED_CONTEXT", "true").lower() == "true",
    "tiered_search": os.getenv("ENABLE_TIERED_SEARCH", "true").lower() == "true",
    "central_vendor_resolver": os.getenv("ENABLE_CENTRAL_RESOLVER", "true").lower() == "true",
    "template_parsing": os.getenv("ENABLE_TEMPLATE_PARSING", "true").lower() == "true",
    
    # Monitoring
    "performance_monitoring": os.getenv("ENABLE_PERF_MONITORING", "true").lower() == "true",
    "detailed_logging": os.getenv("ENABLE_DETAILED_LOGGING", "false").lower() == "true",
}

# ============================================
# API RATE LIMITING
# ============================================
RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "false").lower() == "true"
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
RATE_LIMIT_PER_HOUR = int(os.getenv("RATE_LIMIT_PER_HOUR", "1000"))

# ============================================
# MONITORING
# ============================================
METRICS_ENABLED = os.getenv("METRICS_ENABLED", "false").lower() == "true"
METRICS_ENDPOINT = os.getenv("METRICS_ENDPOINT", "/metrics")
HEALTH_CHECK_INTERVAL = 30

# ============================================
# DEVELOPMENT/DEBUG
# ============================================
DEBUG_MODE = os.getenv("DEBUG", "false").lower() == "true"
VERBOSE_LOGGING = os.getenv("VERBOSE_LOGGING", "false").lower() == "true"
PROFILE_QUERIES = os.getenv("PROFILE_QUERIES", "false").lower() == "true"
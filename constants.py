"""
constants.py - Shared constants for the LLM-Enhanced Procurement RAG system
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
CSV_PATH = "data/Purchase_Orders_and_Contracts.csv"

# ============================================
# CACHE CONFIGURATION
# ============================================
CACHE_MAX_SIZE = 1000
CACHE_TTL_SECONDS = 3600

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

# --- MODIFIED: LLM Model Selection for Specialized Tasks ---
# Use a smaller, faster model for classification, decomposition, and analysis.
DECOMPOSER_MODEL = os.getenv("DECOMPOSER_MODEL", "ibm/granite-3-3-8b-instruct")

# Use a larger, more powerful model for generating final user-facing answers.
SYNTHESIS_MODEL = os.getenv("SYNTHESIS_MODEL", "ibm/granite-3-3-8b-instruct")


LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "800"))
LLM_TOP_P = float(os.getenv("LLM_TOP_P", "0.95"))

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
    "recommendation": 1500,  # Added for LLM recommendations
    "complex_analytical": 2000,  # Added for complex queries
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
    "DELL": ["DELL", "DELL INC", "DELL TECHNOLOGIES", "DELL COMPUTER", "DELL COMPUTERS", "DELL EMC"],
    "IBM": ["IBM", "INTERNATIONAL BUSINESS MACHINES", "IBM CORPORATION", "I.B.M.", "IBM CORP", "IBM GLOBAL"],
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

# Recommendation Keywords (NEW)
RECOMMENDATION_KEYWORDS = [
    'should', 'recommend', 'suggest', 'advice', 'optimize',
    'improve', 'enhance', 'better', 'strategy', 'plan',
    'consider', 'evaluate', 'assess', 'review'
]

# Trend Keywords (NEW)
TREND_KEYWORDS = [
    'trend', 'over time', 'monthly', 'yearly', 'quarterly',
    'growth', 'decline', 'pattern', 'change', 'evolution',
    'trajectory', 'forecast', 'prediction'
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
# LLM PROMPT TEMPLATES (NEW)
# ============================================
# Temperature settings for different query types
LLM_TEMPERATURES = {
    "factual": 0.1,      # Very deterministic for facts
    "analytical": 0.3,   # Some creativity for analysis
    "creative": 0.7,     # More creative for recommendations
    "conversational": 0.5  # Balanced for chat
}

# Max tokens for different response types
LLM_MAX_TOKENS_BY_TYPE = {
    "brief": 200,
    "standard": 500,
    "detailed": 1000,
    "comprehensive": 2000
}

# ============================================
# SEMANTIC SEARCH CONFIGURATION (NEW)
# ============================================
SEMANTIC_SEARCH_TOP_K = 50  # Number of results to retrieve
SEMANTIC_RELEVANCE_THRESHOLD = 0.7  # Minimum relevance score
SEMANTIC_CACHE_ENABLED = True
SEMANTIC_CACHE_TTL = 1800  # 30 minutes

# ============================================
# CONFIDENCE THRESHOLDS (NEW)
# ============================================
HIGH_CONFIDENCE_THRESHOLD = 80
MEDIUM_CONFIDENCE_THRESHOLD = 50
LOW_CONFIDENCE_THRESHOLD = 30

# Use LLM for queries below this confidence
LLM_FALLBACK_CONFIDENCE = 60

# ============================================
# BUSINESS RULES (NEW)
# ============================================
# Thresholds for recommendations
MIN_ORDERS_FOR_VENDOR_ANALYSIS = 5
HIGH_CONCENTRATION_THRESHOLD = 30  # % of total spending
LOW_ACTIVITY_THRESHOLD = 5  # orders
OUTLIER_THRESHOLD_MULTIPLIER = 3  # Standard deviations

# ============================================
# REPORT GENERATION (NEW)
# ============================================
REPORT_FORMATS = ["executive", "detailed", "summary", "technical"]
DEFAULT_REPORT_FORMAT = "executive"
MAX_REPORT_SECTIONS = 10
REPORT_CHART_TYPES = ["bar", "pie", "line", "scatter", "heatmap"]

# ============================================
# ERROR MESSAGES (NEW)
# ============================================
ERROR_MESSAGES = {
    "llm_unavailable": "Advanced AI features are currently unavailable. Using standard processing.",
    "database_error": "Unable to access procurement database. Please try again later.",
    "vendor_not_found": "Vendor not found. Try using the full vendor name or check spelling.",
    "invalid_query": "Query format not recognized. Please try rephrasing in natural language.",
    "timeout": "Query processing timed out. Try simplifying your question.",
}

# ============================================
# FEATURE FLAGS (NEW)
# ============================================
FEATURES = {
    "llm_enabled": os.getenv("ENABLE_LLM", "true").lower() == "true",
    "semantic_search": os.getenv("ENABLE_SEMANTIC", "true").lower() == "true",
    "advanced_analytics": os.getenv("ENABLE_ADVANCED", "true").lower() == "true",
    "caching": os.getenv("ENABLE_CACHE", "true").lower() == "true",
    "parallel_processing": os.getenv("ENABLE_PARALLEL", "true").lower() == "true",
    "natural_language": os.getenv("ENABLE_NL", "true").lower() == "true",
}

# ============================================
# API RATE LIMITING (NEW)
# ============================================
RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "false").lower() == "true"
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
RATE_LIMIT_PER_HOUR = int(os.getenv("RATE_LIMIT_PER_HOUR", "1000"))

# ============================================
# MONITORING (NEW)
# ============================================
METRICS_ENABLED = os.getenv("METRICS_ENABLED", "false").lower() == "true"
METRICS_ENDPOINT = os.getenv("METRICS_ENDPOINT", "/metrics")
HEALTH_CHECK_INTERVAL = 30  # seconds

# ============================================
# DEVELOPMENT/DEBUG
# ============================================
DEBUG_MODE = os.getenv("DEBUG", "false").lower() == "true"
VERBOSE_LOGGING = os.getenv("VERBOSE_LOGGING", "false").lower() == "true"
PROFILE_QUERIES = os.getenv("PROFILE_QUERIES", "false").lower() == "true"
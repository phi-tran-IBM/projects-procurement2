# discovery_store.py - Simple IBM Cloud Elasticsearch connection

from dotenv import load_dotenv
load_dotenv()

import os
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
import urllib3
urllib3.disable_warnings()

import logging

logger = logging.getLogger(__name__)

class DiscoveryStore:
    def __init__(self):
        es_url = os.getenv('DISCOVERY_URL')
        username = os.getenv('DISCOVERY_USERNAME')
        password = os.getenv('DISCOVERY_PASSWORD')

        if not es_url:
            raise ValueError("DISCOVERY_URL environment variable is required")
        
        if not username or not password:
            raise ValueError("DISCOVERY_USERNAME and DISCOVERY_PASSWORD environment variables are required")

        # Remove trailing slash if present (can cause 400 errors)
        es_url = es_url.rstrip('/')
        
        logger.info(f"Connecting to IBM Cloud Elasticsearch at {es_url}")
        logger.info(f"Using credentials for user: {username}")

        try:
            # Simplest possible connection - no headers, just basic auth
            self.es = Elasticsearch(
                es_url,
                basic_auth=(username, password),
                verify_certs=False,  # SSL verification disabled
                ssl_show_warn=False
                # NO HEADERS - they seem to cause 400 errors
            )
            
            logger.info("Testing connection with ping...")
            ping_result = self.es.ping()
            
            if not ping_result:
                raise ConnectionError(f"Ping failed - service may not be running")

            # Test a simple info call
            logger.info("Testing cluster info...")
            info = self.es.info()
            logger.info(f"Connected to Elasticsearch version: {info.get('version', {}).get('number', 'unknown')}")

            # Initialize sentence transformer
            logger.info("Loading sentence transformer model...")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.index = 'procurement'
            
            logger.info("✅ Successfully connected to IBM Cloud Elasticsearch")

        except Exception as e:
            logger.error(f"❌ Connection failed: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            
            # Check if it's a 400 error specifically
            if hasattr(e, 'status_code') and e.status_code == 400:
                logger.error("400 Bad Request - This may indicate:")
                logger.error("  1. Wrong endpoint URL format")
                logger.error("  2. Elasticsearch service not properly configured")
                logger.error("  3. Elasticsearch version compatibility issues")
                
            raise ConnectionError(f"Could not connect to Elasticsearch at {es_url}. Error: {e}")

    def query(self, query_texts, n_results=25):
        """
        Query the Elasticsearch index using semantic search
        """
        if not query_texts or not query_texts[0]:
            logger.warning("Empty query provided")
            return {'metadatas': [[]]}
        
        query_text = query_texts[0]
        logger.debug(f"Processing query: {query_text}")
        
        try:
            # Generate embedding for the query
            embedding = self.model.encode(query_text)
            
            # Simple match_all query first to test basic connectivity
            simple_query = {"match_all": {}}
            
            try:
                # Test with simple query first
                response = self.es.search(
                    index=self.index,
                    size=n_results,
                    query=simple_query
                )
                logger.info(f"Simple query successful - {len(response['hits']['hits'])} results")
                
                # If simple query works, try semantic search
                semantic_query = {
                    "script_score": {
                        "query": {"match_all": {}},
                        "script": {
                            "source": "cosineSimilarity(params.vector, 'embedding') + 1.0",
                            "params": {"vector": embedding.tolist()}
                        }
                    }
                }
                
                response = self.es.search(
                    index=self.index,
                    size=n_results,
                    query=semantic_query
                )
                
            except Exception as semantic_error:
                logger.warning(f"Semantic search failed, falling back to simple search: {semantic_error}")
                # Fall back to simple search if semantic fails
                response = self.es.search(
                    index=self.index,
                    size=n_results,
                    query=simple_query
                )
            
            # Extract source documents from hits
            metadatas = [[hit['_source'] for hit in response['hits']['hits']]]
            
            logger.info(f"Query returned {len(response['hits']['hits'])} results")
            return {'metadatas': metadatas}
            
        except Exception as e:
            logger.error(f"Elasticsearch query failed: {e}")
            raise

    def test_connection(self):
        """Test the connection with detailed diagnostics"""
        try:
            logger.info("=== Connection Diagnostic Test ===")
            
            # Test 1: Basic ping
            logger.info("Test 1: Ping")
            ping_result = self.es.ping()
            logger.info(f"✅ Ping result: {ping_result}")
            
            # Test 2: Cluster info
            logger.info("Test 2: Cluster info")
            info = self.es.info()
            logger.info(f"✅ Cluster name: {info.get('cluster_name', 'unknown')}")
            logger.info(f"✅ ES Version: {info.get('version', {}).get('number', 'unknown')}")
            
            # Test 3: List indices
            logger.info("Test 3: List indices")
            indices = self.es.indices.get_alias(index="*")
            logger.info(f"✅ Found {len(indices)} indices: {list(indices.keys())[:5]}...")  # Show first 5
            
            # Test 4: Check if our index exists
            logger.info(f"Test 4: Check if '{self.index}' index exists")
            exists = self.es.indices.exists(index=self.index)
            logger.info(f"✅ Index '{self.index}' exists: {exists}")
            
            logger.info("=== All tests passed! ===")
            return True
            
        except Exception as e:
            logger.error(f"❌ Diagnostic test failed: {e}")
            return False

# Create the global collection instance
collection = DiscoveryStore()
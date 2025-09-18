# discovery_store.py

from dotenv import load_dotenv
load_dotenv()

import os
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
import urllib3
urllib3.disable_warnings()

import logging

logger = logging.getLogger(__name__)

class MockElasticsearch:
    """A mock Elasticsearch client to be used when credentials are not available."""
    def ping(self):
        logger.info("MockElasticsearch: ping successful.")
        return True

    def search(self, index, size, query):
        logger.info(f"MockElasticsearch: Searching index '{index}' with query: {query}")
        # Return a predictable, structured response that mimics the real API
        return {
            'hits': {
                'hits': [
                    {'_source': {'VENDOR_NAME_1': 'MOCK VENDOR', 'ITEM_TOTAL_COST': 123.45, 'ITEM_DESCRIPTION': 'Mock item description for testing.'}}
                ]
            }
        }

class DiscoveryStore:
    def __init__(self):
        es_url = os.getenv('DISCOVERY_URL')

        if not es_url:
            logger.warning("DISCOVERY_URL not set. Using MockElasticsearch for testing purposes.")
            self.es = MockElasticsearch()
            self.model = None  # No model needed for mock
            self.index = 'procurement'
            return

        username = os.getenv('DISCOVERY_USERNAME', 'admin')
        password = os.getenv('DISCOVERY_PASSWORD')
        headers = {
            "Accept": "application/vnd.elasticsearch+json; compatible-with=8",
            "Content-Type": "application/vnd.elasticsearch+json; compatible-with=8"
        }

        try:
            self.es = Elasticsearch(
                es_url,
                basic_auth=(username, password),
                verify_certs=False,
                ssl_show_warn=False,
                headers=headers
            )
            if not self.es.ping():
                raise ConnectionError(f"Could not connect to Elasticsearch at {es_url}")

            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.index = 'procurement'
            logger.info(f"Successfully connected to Elasticsearch at {es_url}")

        except Exception as e:
            logger.error(f"Failed to initialize real Elasticsearch client: {e}. Using MockElasticsearch.")
            self.es = MockElasticsearch()
            self.model = None
            self.index = 'procurement'
    
    def query(self, query_texts, n_results=25):
        # The mock is handled transparently by the search method now.
        # If self.es is the mock, its search method will be called.
        if not self.es or not query_texts or not query_texts[0]:
            return {'metadatas': [[]]}
        
        # If using the real ES, we need the model.
        if isinstance(self.es, Elasticsearch) and not self.model:
             logger.error("Real Elasticsearch client is present but sentence transformer model is not. Cannot create embeddings.")
             return {'metadatas': [[]]}

        try:
            # For real ES, encode the query. For mock, this is not used but doesn't hurt.
            if self.model:
                embedding = self.model.encode(query_texts[0])
                es_query = {
                    "script_score": {
                        "query": {"match_all": {}},
                        "script": {
                            "source": "cosineSimilarity(params.vector, 'embedding') + 1.0",
                            "params": {"vector": embedding.tolist()}
                        }
                    }
                }
            else:
                # Mock search doesn't need a real query
                es_query = {"match_all": {}}

            response = self.es.search(
                index=self.index,
                size=n_results,
                query=es_query
            )

            metadatas = [[hit['_source'] for hit in response['hits']['hits']]]
            return {'metadatas': metadatas}
        except Exception as e:
            logger.error(f"Elasticsearch query failed: {e}. Returning empty results.")
            return {'metadatas': [[]]}

collection = DiscoveryStore()
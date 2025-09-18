# discovery_store.py

from dotenv import load_dotenv
load_dotenv()

import os
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
import urllib3
urllib3.disable_warnings()

class DiscoveryStore:
    def __init__(self):
        es_url = os.getenv('DISCOVERY_URL')
        username = os.getenv('DISCOVERY_USERNAME', 'admin')
        password = os.getenv('DISCOVERY_PASSWORD')
        
        # FIX: Define headers for Elasticsearch 8 compatibility
        # This resolves the error: "Accept version must be either version 8 or 7, but found 9"
        headers = {
            "Accept": "application/vnd.elasticsearch+json; compatible-with=8",
            "Content-Type": "application/vnd.elasticsearch+json; compatible-with=8"
        }

        self.es = Elasticsearch(
            es_url,
            basic_auth=(username, password),
            verify_certs=False,
            ssl_show_warn=False,
            headers=headers # Apply compatibility headers
        )
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = 'procurement'
    
    def query(self, query_texts, n_results=25):
        if not query_texts or not query_texts[0]:
            return {'metadatas': [[]]}
        
        embedding = self.model.encode(query_texts[0])
        
        response = self.es.search(
            index=self.index,
            size=n_results,
            query={
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.vector, 'embedding') + 1.0",
                        "params": {"vector": embedding.tolist()}
                    }
                }
            }
        )
        
        metadatas = [[hit['_source'] for hit in response['hits']['hits']]]
        return {'metadatas': metadatas}

collection = DiscoveryStore()
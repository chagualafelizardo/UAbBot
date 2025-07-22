from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
import os

class ActionSearchDocuments(Action):
    def __init__(self):
        # Carrega os embeddings
        with open('/app/data/embeddings.pkl', 'rb') as f:
            self.data = pickle.load(f)
        
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.mongo_uri = os.getenv("MONGO_URI", "mongodb://root:root@mongodb:27017/")
        self.db_name = os.getenv("DB_NAME", "uab")

    def name(self) -> Text:
        return "action_search_documents"

    async def run(self, dispatcher: CollectingDispatcher,
                 tracker: Tracker,
                 domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        query = tracker.latest_message.get('text')
        query_embedding = self.model.encode(query)
        
        # Calcula similaridade de cosseno
        scores = np.dot(self.data['embeddings'], query_embedding.T)
        top_idx = np.argmax(scores)
        best_match = self.data['documents'][top_idx]
        
        # Recupera o documento completo do MongoDB se necessário
        client = MongoClient(self.mongo_uri)
        db = client[self.db_name]
        doc = db.documents.find_one({"filename": best_match['filename']})
        client.close()
        
        # Formata a resposta
        response = (
            f"Encontrei esta informação no documento '{best_match['filename']}':\n\n"
            f"{best_match['text'][:500]}..."  # Limita o tamanho da resposta
        )
        
        dispatcher.utter_message(text=response)
        return []
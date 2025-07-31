import re
from typing import List, Dict
from rapidfuzz import fuzz, process

class UniversalDocumentSearcher:
    def __init__(self):
        self.client = MongoClient(MONGO_URI)
        self.db = self.client[DB_NAME]
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def find_most_relevant(self, query: str, top_n: int = 3) -> List[Dict]:
        """Busca universal em todos os documentos"""
        try:
            # Pré-processamento da query
            clean_query = self._preprocess_text(query)
            
            # Busca todos os documentos
            all_docs = list(self.db.documents.find({}))
            
            # Classificação por relevância
            results = []
            for doc in all_docs:
                text = doc.get('text_content', '')
                score = self._calculate_relevance(clean_query, text)
                
                if score > 0.3:  # Threshold mínimo
                    results.append({
                        'text': self._extract_best_match(clean_query, text),
                        'source': doc.get('filename', 'Documento'),
                        'score': score
                    })
            
            # Ordena e filtra os melhores
            results.sort(key=lambda x: x['score'], reverse=True)
            return results[:top_n]
            
        except Exception as e:
            logger.error(f"Erro na busca universal: {str(e)}")
            return []

    def _preprocess_text(self, text: str) -> str:
        """Limpa e normaliza o texto"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text.strip()

    def _calculate_relevance(self, query: str, text: str) -> float:
        """Calcula a relevância combinando várias métricas"""
        # Similaridade semântica
        query_embedding = self.model.encode(query)
        text_embedding = self.model.encode(text)
        semantic_sim = cosine_similarity(
            query_embedding.reshape(1, -1),
            text_embedding.reshape(1, -1)
        )[0][0]
        
        # Similaridade textual
        text_sim = fuzz.token_set_ratio(query, text) / 100
        
        # Combina as métricas (70% semântica, 30% textual)
        return 0.7 * semantic_sim + 0.3 * text_sim

    def _extract_best_match(self, query: str, text: str) -> str:
        """Extrai a parte mais relevante do texto"""
        sentences = re.split(r'[.!?]', text)
        if not sentences:
            return text[:500] + ('...' if len(text) > 500 else '')
            
        best_sentence = max(
            sentences,
            key=lambda x: fuzz.token_set_ratio(query, x),
            default=''
        )
        
        return best_sentence.strip()[:500] + ('...' if len(best_sentence) > 500 else '')
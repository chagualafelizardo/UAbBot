from rasa_sdk import Action
import pymongo
import logging
from pymongo.errors import PyMongoError
import re
from typing import Dict, List, Optional
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class ActionSmartSearch(Action):
    def name(self):
        return "action_search_documents"

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.min_similarity = 0.3  # Threshold mínimo de similaridade
        
        try:
            # Conexão com MongoDB
            self.client = pymongo.MongoClient(
                "mongodb://root:root@uabbot-mongodb-1:27017/",
                serverSelectionTimeoutMS=5000
            )
            self.db = self.client["uab"]
            self.collection = self.db["documents"]
            
            # Modelo leve para embeddings
            self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            
            # Verificar conexão
            self.client.admin.command('ping')
            count = self.collection.count_documents({})
            self.logger.info(f"Conexão OK. Coleção possui {count} documentos.")
            
            # Criar índices
            self._create_indexes()
                
        except Exception as e:
            self.logger.error(f"Erro na inicialização: {str(e)}")
            raise

    def _create_indexes(self):
        """Cria índices necessários"""
        try:
            if "text_content_text" not in self.collection.index_information():
                self.collection.create_index(
                    [("text_content", "text")],
                    default_language="portuguese"
                )
        except Exception as e:
            self.logger.error(f"Erro ao criar índices: {str(e)}")

    def _get_embedding(self, text: str) -> List[float]:
        """Gera embedding para um texto"""
        return self.model.encode(text, convert_to_tensor=False).tolist()

    def _extract_faqs(self, content: str) -> List[Dict]:
        """Extrai FAQs formatadas do conteúdo"""
        faqs = []
        for match in re.finditer(r'(\d+\.\s)?(.*?\?)(.*?)(?=\d+\.\s|\Z)', content, re.DOTALL):
            faqs.append({
                'question': match.group(2).strip(),
                'answer': match.group(3).strip(),
                'type': 'faq'
            })
        return faqs

    def _extract_courses(self, content: str) -> List[Dict]:
        """Extrai informações de cursos do conteúdo"""
        courses = []
        pattern = r'(?P<type>Licenciatura|Mestrado|Doutoramento)[\s:-]+(?P<name>[^\n]+)(?P<details>[\s\S]+?)(?=(Licenciatura|Mestrado|Doutoramento|$))'
        
        for match in re.finditer(pattern, content, re.IGNORECASE):
            courses.append({
                'type': match.group('type'),
                'name': match.group('name').strip(),
                'details': ' '.join(match.group('details').split()[:30]),
                'content_type': 'course'
            })
        return courses

    def _find_best_match(self, query: str, docs: List[Dict]) -> Optional[Dict]:
        """Encontra a melhor correspondência para a consulta"""
        query_embedding = self._get_embedding(query)
        best_match = None
        highest_score = 0
        
        for doc in docs:
            content = doc.get("text_content", "")
            
            # Processa FAQs
            faqs = self._extract_faqs(content)
            for faq in faqs:
                similarity = cosine_similarity(
                    [query_embedding],
                    [self._get_embedding(faq['question'])]
                )[0][0]
                
                if similarity > highest_score and similarity >= self.min_similarity:
                    highest_score = similarity
                    best_match = {
                        'type': 'faq',
                        'question': faq['question'],
                        'answer': faq['answer'],
                        'score': similarity,
                        'filename': doc.get("filename", "documento")
                    }
            
            # Processa cursos se for uma pergunta sobre cursos
            if any(word in query.lower() for word in ["curso", "licenciatura", "mestrado", "doutoramento"]):
                courses = self._extract_courses(content)
                for course in courses:
                    course_text = f"{course['type']} {course['name']}"
                    similarity = cosine_similarity(
                        [query_embedding],
                        [self._get_embedding(course_text)]
                    )[0][0]
                    
                    if similarity > highest_score and similarity >= self.min_similarity:
                        highest_score = similarity
                        best_match = {
                            'type': 'course',
                            'title': course_text,
                            'details': course['details'],
                            'score': similarity,
                            'filename': doc.get("filename", "documento")
                        }
        
        return best_match

    def _format_response(self, match: Dict) -> str:
        """Formata a resposta de acordo com o tipo de conteúdo"""
        if match['type'] == 'faq':
            return (
                f"📚 FAQ encontrada em {match['filename']}:\n"
                f"❓ {match['question']}\n\n"
                f"💡 {match['answer'][:500]}{'...' if len(match['answer']) > 500 else ''}"
            )
        elif match['type'] == 'course':
            return (
                f"🎓 Informação sobre cursos em {match['filename']}:\n"
                f"📌 {match['title']}\n\n"
                f"ℹ️ {match['details'][:500]}{'...' if len(match['details']) > 500 else ''}"
            )
        return ""

    def run(self, dispatcher, tracker, domain):
        query = tracker.latest_message.get('text')
        self.logger.info(f"Processando consulta: '{query}'")
        
        try:
            # Busca textual inicial
            docs = list(self.collection.find(
                {"$text": {"$search": query}},
                {"score": {"$meta": "textScore"}, "text_content": 1, "filename": 1}
            ).sort([("score", {"$meta": "textScore"})]).limit(3))
            
            if not docs:
                dispatcher.utter_message(text="Não encontrei informações sobre esse tópico.")
                return []
            
            # Encontra a melhor correspondência
            best_match = self._find_best_match(query, docs)
            
            if best_match:
                response = self._format_response(best_match)
                dispatcher.utter_message(text=response)
            else:
                # Fallback: mostra trecho mais relevante
                best_doc = max(docs, key=lambda x: x['score'])
                sentences = re.split(r'(?<=[.!?])\s+', best_doc['text_content'])
                if sentences:
                    best_sentence = max(sentences, key=lambda x: len(x))[:400]
                    dispatcher.utter_message(
                        text=f"📄 {best_doc.get('filename', 'documento')}\n\n"
                             f"🔍 Trecho relevante:\n{best_sentence}{'...' if len(best_sentence) == 400 else ''}"
                    )
                else:
                    dispatcher.utter_message(text="Encontrei documentos relacionados, mas não informações específicas para sua pergunta.")
                
        except Exception as e:
            self.logger.error(f"Erro na busca: {str(e)}", exc_info=True)
            dispatcher.utter_message(text="Desculpe, ocorreu um erro ao processar sua solicitação.")

        return []
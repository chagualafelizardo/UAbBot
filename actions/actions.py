from rasa_sdk import Action
import pymongo
import logging
from pymongo.errors import PyMongoError
import re
from typing import Dict, List, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import threading

class ActionSmartSearch(Action):
    def name(self):
        return "action_search_documents"

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.model_ready = False
        self.min_similarity = 0.35
        
        # Inicialização rápida do MongoDB
        try:
            self.client = pymongo.MongoClient(
                "mongodb://root:root@uabbot-mongodb-1:27017/",
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=10000,
                socketTimeoutMS=10000
            )
            self.db = self.client["uab"]
            self.collection = self.db["documents"]
            self.client.admin.command('ping')
            self.logger.info("Conexão MongoDB estabelecida")
            
            # Cria índices em segundo plano
            threading.Thread(target=self._create_indexes, daemon=True).start()
            
            # Carrega o modelo em segundo plano
            threading.Thread(target=self._load_model, daemon=True).start()
            
        except Exception as e:
            self.logger.error(f"Erro na inicialização: {str(e)}")
            raise

    def _load_model(self):
        """Carrega o modelo em segundo plano"""
        try:
            from sentence_transformers import SentenceTransformer
            self.logger.info("Iniciando carregamento do modelo...")
            
            # Modelo leve e rápido (1/4 do tamanho do anterior)
            self.model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
            self.model.max_seq_length = 128  # Reduz ainda mais para performance
            self.model_ready = True
            self.logger.info("Modelo carregado com sucesso")
            
        except Exception as e:
            self.logger.error(f"Erro ao carregar modelo: {str(e)}")

    def _create_indexes(self):
        """Cria índices em segundo plano"""
        try:
            if "text_content_text" not in self.collection.index_information():
                self.collection.create_index(
                    [("text_content", "text")],
                    default_language="portuguese"
                )
                self.logger.info("Índice de texto criado")
        except Exception as e:
            self.logger.error(f"Erro ao criar índices: {str(e)}")

    def _simple_search(self, query: str, docs: List[Dict]) -> Optional[Dict]:
        """Busca textual simples enquanto o modelo não está pronto"""
        best_match = None
        highest_score = 0
        query_lower = query.lower()
        
        for doc in docs:
            content = doc.get("text_content", "").lower()
            
            # Verifica correspondência direta de palavras-chave
            score = sum(1 for word in query_lower.split() if word in content) / len(query.split())
            
            if score > highest_score:
                highest_score = score
                best_match = doc
                
        return best_match if highest_score > 0.3 else None

    def _extract_faq(self, content: str, query: str) -> Optional[Dict]:
        """Extrai FAQ relevante usando regex simples"""
        best_faq = None
        highest_score = 0
        
        for match in re.finditer(r'(\d+\.\s)?(.*?\?)(.*?)(?=\d+\.\s|\Z)', content, re.DOTALL):
            question = match.group(2).strip()
            answer = match.group(3).strip()
            
            # Similaridade simples baseada em palavras-chave
            score = sum(1 for word in query.lower().split() if word in question.lower()) / len(query.split())
            
            if score > highest_score:
                highest_score = score
                best_faq = {
                    'question': question,
                    'answer': answer,
                    'score': score
                }
        
        return best_faq if highest_score > 0.3 else None

    def _extract_course_info(self, content: str) -> List[Dict]:
        """Extrai informações de cursos com regex otimizado"""
        courses = []
        for match in re.finditer(r'(Licenciatura|Mestrado|Doutoramento)[\s:-]+([^\n]+)', content, re.IGNORECASE):
            courses.append({
                'type': match.group(1),
                'name': match.group(2).strip(),
                'content': content[match.end():match.end()+300]  # Limita o conteúdo
            })
        return courses

    def run(self, dispatcher, tracker, domain):
        query = tracker.latest_message.get('text')
        self.logger.info(f"Processando consulta: '{query}'")
        
        try:
            # Busca textual inicial (rápida)
            docs = list(self.collection.find(
                {"$text": {"$search": query}},
                {"score": {"$meta": "textScore"}, "text_content": 1, "filename": 1}
            ).sort([("score", {"$meta": "textScore"})]).limit(3))
            
            if not docs:
                dispatcher.utter_message(text="Não encontrei informações sobre esse tópico.")
                return []
            
            # Se o modelo ainda não está pronto, usa busca simples
            if not self.model_ready:
                self.logger.warning("Modelo não carregado ainda, usando busca simples")
                best_doc = self._simple_search(query, docs)
                
                if best_doc:
                    # Tenta extrair FAQ primeiro
                    faq = self._extract_faq(best_doc['text_content'], query)
                    if faq:
                        response = f"📄 {best_doc.get('filename', 'documento')}\n\n"
                        response += f"❓ {faq['question']}\n💡 {faq['answer'][:300]}..."
                        dispatcher.utter_message(text=response)
                        return []
                    
                    # Fallback para cursos se a pergunta for sobre cursos
                    if any(w in query.lower() for w in ["curso", "licenciatura", "mestrado", "doutoramento"]):
                        courses = self._extract_course_info(best_doc['text_content'])
                        if courses:
                            course = courses[0]
                            response = f"🎓 {course['type']} em {course['name']}\n"
                            response += f"📄 {best_doc.get('filename', 'documento')}\n"
                            response += f"ℹ️ {course['content'][:300]}..."
                            dispatcher.utter_message(text=response)
                            return []
                    
                    # Fallback final: trecho mais relevante
                    sentences = re.split(r'(?<=[.!?])\s+', best_doc['text_content'])
                    if sentences:
                        best_sentence = max(sentences, key=lambda x: len(x))[:300]
                        dispatcher.utter_message(
                            text=f"📄 {best_doc.get('filename', 'documento')}\n\n"
                                 f"🔍 Trecho relevante:\n{best_sentence}..."
                        )
                        return []
                
                dispatcher.utter_message(text="Encontrei documentos relacionados, mas não informações específicas para sua pergunta.")
                return []
            
            # Se o modelo está pronto, usa busca semântica
            query_embedding = self.model.encode(query)
            
            best_match = None
            highest_score = 0
            
            for doc in docs:
                content = doc.get("text_content", "")
                
                # Gera embedding sob demanda (com cache implícito)
                doc_embedding = self.model.encode(content[:1000])  # Limita tamanho
                
                # Calcula similaridade
                similarity = cosine_similarity([query_embedding], [doc_embedding])[0][0]
                
                if similarity > highest_score:
                    highest_score = similarity
                    best_match = {
                        'doc': doc,
                        'score': similarity
                    }
            
            if best_match and highest_score > self.min_similarity:
                doc = best_match['doc']
                content = doc['text_content']
                
                # Prioriza FAQs
                faq = self._extract_faq(content, query)
                if faq:
                    response = f"📄 {doc.get('filename', 'documento')}\n\n"
                    response += f"❓ {faq['question']}\n💡 {faq['answer'][:300]}..."
                    dispatcher.utter_message(text=response)
                    return []
                
                # Busca por cursos se relevante
                if any(w in query.lower() for w in ["curso", "licenciatura", "mestrado", "doutoramento"]):
                    courses = self._extract_course_info(content)
                    if courses:
                        course = courses[0]
                        response = f"🎓 {course['type']} em {course['name']}\n"
                        response += f"📄 {doc.get('filename', 'documento')}\n"
                        response += f"ℹ️ {course['content'][:300]}..."
                        dispatcher.utter_message(text=response)
                        return []
                
                # Fallback para trecho relevante
                sentences = re.split(r'(?<=[.!?])\s+', content)
                if sentences:
                    best_sentence = max(sentences, key=lambda x: len(x))[:300]
                    dispatcher.utter_message(
                        text=f"📄 {doc.get('filename', 'documento')}\n\n"
                             f"🔍 Trecho relevante:\n{best_sentence}..."
                    )
                    return []
            
            dispatcher.utter_message(text="Encontrei documentos relacionados, mas não informações específicas para sua pergunta.")
            
        except Exception as e:
            self.logger.error(f"Erro na busca: {str(e)}", exc_info=True)
            dispatcher.utter_message(text="Desculpe, ocorreu um erro ao processar sua solicitação.")

        return []
from rasa_sdk import Action
import pymongo
import logging
from pymongo.errors import PyMongoError
import re
from typing import Optional, Dict

class ActionSmartSearch(Action):
    def name(self):
        return "action_search_documents"

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        try:
            # Conexão com MongoDB
            self.client = pymongo.MongoClient(
                "mongodb://root:root@uabbot-mongodb-1:27017/",
                serverSelectionTimeoutMS=5000
            )
            self.db = self.client["uab"]
            self.collection = self.db["documents"]
            
            # Verificar conexão
            self.client.admin.command('ping')
            count = self.collection.count_documents({})
            self.logger.info(f"Conexão OK. Coleção possui {count} documentos.")
            
            # Criar índice de texto composto
            self._create_text_index()
                
        except PyMongoError as e:
            self.logger.error(f"Erro MongoDB: {str(e)}")
            raise

    def _create_text_index(self):
        """Cria um índice de texto composto de forma segura"""
        try:
            # Verifica se já existe algum índice de texto
            existing_indexes = self.collection.index_information()
            text_index_exists = any(
                any('text' in str(field[1]) for field in idx_info['key'])
                for idx_info in existing_indexes.values()
            )
            
            if not text_index_exists:
                # Cria índice composto com ambos os campos
                self.collection.create_index([
                    ("text_content", "text"),
                    ("filename", "text")
                ], default_language="portuguese")
                self.logger.info("Índice de texto composto criado com sucesso")
            else:
                self.logger.info("Índice de texto já existe, mantendo o atual")
                
        except Exception as e:
            self.logger.error(f"Erro ao verificar/criar índices: {str(e)}")
            # Continua mesmo sem índice para não bloquear a execução

    def identify_content_type(self, filename: str, content: str) -> str:
        """Identifica o tipo de conteúdo baseado no nome do arquivo e conteúdo"""
        if "faq" in filename.lower() or "perguntas frequentes" in content[:200].lower():
            return "faq"
        elif "curso" in filename.lower() or "licenciatura" in content.lower() or "mestrado" in content.lower():
            return "curso"
        elif "regulamento" in filename.lower() or "normas" in content.lower():
            return "regulamento"
        elif "admissão" in filename.lower() or "candidatura" in content.lower():
            return "admissao"
        else:
            return "geral"

    def extract_faq_answer(self, content: str, query: str) -> Optional[Dict]:
        """Extrai resposta de FAQ formatada"""
        for match in re.finditer(r'(\d+\.\s)(.*?\?)(.*?)(?=\d+\.\s|\Z)', content, re.DOTALL):
            question = match.group(2).strip()
            if self.calculate_similarity(question, query) > 0.3:
                return {
                    'type': 'faq',
                    'question': question,
                    'answer': match.group(3).strip()
                }
        return None

    def extract_course_info(self, content: str, query: str) -> Optional[Dict]:
        """Extrai informações sobre cursos"""
        for match in re.finditer(r'(Curso|Licenciatura|Mestrado|Doutoramento)[^\n]+?\n(.+?)(?=\n\s*(Curso|Licenciatura|Mestrado|Doutoramento|\Z))', 
                               content, re.DOTALL | re.IGNORECASE):
            course_title = match.group(0).split('\n')[0].strip()
            if self.calculate_similarity(course_title, query) > 0.3:
                return {
                    'type': 'curso',
                    'title': course_title,
                    'description': match.group(2).strip()
                }
        return None

    def extract_general_info(self, content: str, query: str) -> Optional[Dict]:
        """Extrai trecho relevante de documentos gerais"""
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', content)
        best_sentence = max(sentences, 
                          key=lambda x: self.calculate_similarity(x, query), 
                          default="")
        if self.calculate_similarity(best_sentence, query) > 0.2:
            return {
                'type': 'geral',
                'content': best_sentence
            }
        return None

    def calculate_similarity(self, text: str, query: str) -> float:
        """Calcula similaridade entre texto e consulta"""
        text_words = set(re.sub(r'[^\w\s]', '', text.lower()).split())
        query_words = set(re.sub(r'[^\w\s]', '', query.lower()).split())
        return len(text_words & query_words) / len(query_words) if query_words else 0

    def format_response(self, result: Dict, filename: str) -> str:
        """Formata a resposta conforme o tipo de conteúdo"""
        if result['type'] == 'faq':
            return (f"📚 FAQ encontrada em {filename}:\n"
                   f"❓ {result['question']}\n\n"
                   f"💡 {self.clean_text(result['answer'])}")
        
        elif result['type'] == 'curso':
            return (f"🎓 Informação sobre cursos em {filename}:\n"
                   f"📌 {result['title']}\n\n"
                   f"ℹ️ {self.clean_text(result['description'])}")
        
        else:
            return (f"📄 Documento: {filename}\n\n"
                   f"🔍 Trecho relevante:\n{self.clean_text(result['content'])}")

    def clean_text(self, text: str, max_length: int = 500) -> str:
        """Limpa e formata o texto"""
        cleaned = ' '.join(text.split())
        return cleaned[:max_length] + ("..." if len(cleaned) > max_length else "")

    def run(self, dispatcher, tracker, domain):
        query = tracker.latest_message.get('text')
        self.logger.info(f"Processando consulta: '{query}'")
        
        try:
            # Busca nos documentos (limita a 5 resultados)
            docs = list(self.collection.find(
                {"$text": {"$search": query}},
                {"score": {"$meta": "textScore"}, "text_content": 1, "filename": 1}
            ).sort([("score", {"$meta": "textScore"})]).limit(5))
            
            if not docs:
                dispatcher.utter_message(text="Não encontrei informações sobre esse tópico.")
                return []
            
            responses = []
            for doc in docs:
                content = doc.get("text_content", "")
                filename = doc.get("filename", "documento")
                content_type = self.identify_content_type(filename, content)
                
                result = None
                if content_type == "faq":
                    result = self.extract_faq_answer(content, query)
                elif content_type == "curso":
                    result = self.extract_course_info(content, query)
                else:
                    result = self.extract_general_info(content, query)
                
                if result:
                    responses.append(self.format_response(result, filename))
                    if len(responses) >= 3:  # Limita a 3 respostas
                        break
            
            if responses:
                dispatcher.utter_message(text="\n\n---\n\n".join(responses))
            else:
                dispatcher.utter_message(text="Encontrei documentos relacionados, mas não informações específicas para sua pergunta.")
                
        except Exception as e:
            self.logger.error(f"Erro na busca: {str(e)}", exc_info=True)
            dispatcher.utter_message(text="Desculpe, ocorreu um erro ao processar sua solicitação.")

        return []
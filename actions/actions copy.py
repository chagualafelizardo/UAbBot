# from rasa_sdk import Action
# import pymongo
# import logging
# from pymongo.errors import PyMongoError
# import re

# class ActionSearchFAQs(Action):
#     def name(self):
#         return "action_search_documents"

#     def __init__(self):
#         self.logger = logging.getLogger(__name__)
        
#         try:
#             # Conexão com MongoDB
#             self.client = pymongo.MongoClient(
#                 "mongodb://root:root@uabbot-mongodb-1:27017/",
#                 serverSelectionTimeoutMS=5000
#             )
#             self.db = self.client["uab"]
#             self.collection = self.db["documents"]
            
#             # Verificar conexão
#             self.client.admin.command('ping')
#             count = self.collection.count_documents({})
#             self.logger.info(f"Conexão OK. Coleção possui {count} documentos.")
            
#             # Criar índice de texto se não existir
#             if "text_content_text" not in self.collection.index_information():
#                 self.collection.create_index([("text_content", "text")])
#                 self.logger.info("Índice de texto criado para busca full-text")
                
#         except PyMongoError as e:
#             self.logger.error(f"Erro MongoDB: {str(e)}")
#             raise

#     def find_matching_faq(self, text_content, query):
#         """
#         Encontra a FAQ mais relevante para a consulta e retorna a pergunta e resposta
#         """
#         # Padrão para identificar perguntas e respostas no formato "1. Pergunta? Resposta"
#         faq_pattern = re.compile(
#             r'(\d+\.\s)(.*?\?)(.*?)(?=\d+\.\s|\Z)', 
#             re.DOTALL
#         )
        
#         best_match = None
#         highest_score = 0
        
#         # Normaliza a query para comparação
#         normalized_query = query.lower().strip()
        
#         for match in faq_pattern.finditer(text_content):
#             question = match.group(2).strip()  # A pergunta da FAQ
#             answer = match.group(3).strip()    # A resposta da FAQ
            
#             # Calcula similaridade simples (pode ser melhorado)
#             score = self.calculate_similarity(question, normalized_query)
            
#             if score > highest_score:
#                 highest_score = score
#                 best_match = {
#                     'question': question,
#                     'answer': answer,
#                     'score': score
#                 }
        
#         return best_match

#     def calculate_similarity(self, question, query):
#         """
#         Calcula uma pontuação de similaridade simples entre a pergunta da FAQ e a consulta
#         """
#         # Remove pontuação e normaliza
#         question_words = set(re.sub(r'[^\w\s]', '', question.lower()).split())
#         query_words = set(re.sub(r'[^\w\s]', '', query.lower()).split())
        
#         # Conta palavras em comum
#         common_words = question_words.intersection(query_words)
#         return len(common_words) / len(query_words) if query_words else 0

#     def clean_answer(self, answer):
#         """
#         Limpa a resposta removendo espaços excessivos e quebras de linha
#         """
#         # Remove múltiplos espaços e quebras de linha
#         cleaned = ' '.join(answer.split())
#         # Limita o tamanho da resposta
#         return cleaned[:500] + "..." if len(cleaned) > 500 else cleaned

#     def run(self, dispatcher, tracker, domain):
#         query = tracker.latest_message.get('text')
#         self.logger.info(f"Processando consulta: '{query}'")
        
#         try:
#             # Busca no MongoDB
#             docs = list(self.collection.find(
#                 {"$text": {"$search": query}},
#                 {"score": {"$meta": "textScore"}, "text_content": 1}
#             ).sort([("score", {"$meta": "textScore"})]).limit(1))
            
#             if not docs:
#                 dispatcher.utter_message(text="Não encontrei informações sobre isso nas FAQs.")
#                 return []
            
#             # Encontra a FAQ mais relevante no texto
#             text_content = docs[0].get("text_content", "")
#             faq_match = self.find_matching_faq(text_content, query)
            
#             if faq_match and faq_match['score'] > 0.3:  # Threshold de similaridade
#                 response = f"Pergunta relacionada: {faq_match['question']}\n\n"
#                 response += f"Resposta: {self.clean_answer(faq_match['answer'])}"
#                 dispatcher.utter_message(text=response)
#             else:
#                 dispatcher.utter_message(text="Encontrei informações nas FAQs, mas não uma resposta exata para sua pergunta.")
                
#         except Exception as e:
#             self.logger.error(f"Erro na busca: {str(e)}", exc_info=True)
#             dispatcher.utter_message(text="Desculpe, ocorreu um erro ao buscar nas FAQs.")

#         return []
# from rasa_sdk import Action
# import pymongo
# import logging
# import re
# from typing import Dict, List, Any
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np
# from transformers import RobertaTokenizer, RobertaModel
# import torch
# from datetime import datetime

# logger = logging.getLogger(__name__)

# class ActionSmartSearch(Action):
#     def name(self):
#         return "action_search_documents"

#     def __init__(self):
#         super().__init__()
#         self.logger = logging.getLogger(__name__)
#         self.min_similarity = 0.35
#         self.context_window = 5
#         self.top_k = 3  # Número de documentos a retornar
        
#         # Conexão com MongoDB
#         self.client = pymongo.MongoClient("mongodb://root:root@uabbot-mongodb-1:27017/")
#         self.db = self.client["uab"]
#         self.collection = self.db["documents"]
        
#         # Modelos de embeddings
#         self.sentence_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
#         self.roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
#         self.roberta_model = RobertaModel.from_pretrained('roberta-base')
        
#         # Configuração RAG
#         self.embedding_dim = 768  # Dimensão dos embeddings de RoBERTa
#         self.index = None
#         self._initialize_index()

#     def _initialize_index(self):
#         """Inicializa o índice de busca semântica"""
#         try:
#             # Verificar se já existe uma coleção de embeddings
#             if "document_embeddings" not in self.db.list_collection_names():
#                 self.db.create_collection("document_embeddings")
#                 self.logger.info("Coleção document_embeddings criada com sucesso")
            
#             # Carregar ou gerar embeddings
#             self._generate_document_embeddings()
            
#         except Exception as e:
#             self.logger.error(f"Erro ao inicializar índice: {str(e)}")
#             raise

#     def _generate_document_embeddings(self):
#         """Gera embeddings para todos os documentos se não existirem"""
#         count = self.db.document_embeddings.count_documents({})
#         if count == 0:
#             self.logger.info("Gerando embeddings para documentos...")
#             documents = self.collection.find({})
            
#             batch = []
#             for doc in documents:
#                 content = doc.get("text_content", "")
#                 if not content:
#                     continue
                    
#                 # Gerar embedding com RoBERTa
#                 embedding = self._get_roberta_embedding(content[:512])  # Limitar a 512 tokens
                
#                 batch.append({
#                     "doc_id": doc["_id"],
#                     "filename": doc.get("filename", ""),
#                     "embedding": embedding.tolist(),
#                     "last_updated": datetime.now()
#                 })
                
#                 if len(batch) >= 100:
#                     self.db.document_embeddings.insert_many(batch)
#                     batch = []
            
#             if batch:
#                 self.db.document_embeddings.insert_many(batch)
            
#             self.logger.info(f"Embeddings gerados para {self.db.document_embeddings.count_documents({})} documentos")

#     def _get_roberta_embedding(self, text: str) -> np.ndarray:
#         """Obtém o embedding de um texto usando RoBERTa"""
#         inputs = self.roberta_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
#         with torch.no_grad():
#             outputs = self.roberta_model(**inputs)
        
#         # Usar o embedding do token [CLS] como representação do documento
#         cls_embedding = outputs.last_hidden_state[:, 0, :].numpy()
#         return cls_embedding.squeeze()

#     def _semantic_search(self, query: str, top_k: int = 3) -> List[Dict]:
#         """Realiza busca semântica usando embeddings"""
#         query_embedding = self._get_roberta_embedding(query)
        
#         # Buscar no MongoDB usando busca por similaridade de cosseno
#         pipeline = [
#             {
#                 "$addFields": {
#                     "similarity": {
#                         "$let": {
#                             "vars": {
#                                 "dot_product": {
#                                     "$reduce": {
#                                         "input": {"$zip": {"inputs": ["$embedding", query_embedding.tolist()]}},
#                                         "initialValue": 0,
#                                         "in": {
#                                             "$add": [
#                                                 "$$value",
#                                                 {"$multiply": [{"$arrayElemAt": ["$$this", 0]}, {"$arrayElemAt": ["$$this", 1]}]}
#                                             ]
#                                         }
#                                     }
#                                 },
#                                 "query_norm": {"$sqrt": {"$reduce": {
#                                     "input": query_embedding.tolist(),
#                                     "initialValue": 0,
#                                     "in": {"$add": ["$$value", {"$pow": ["$$this", 2]}]}
#                                 }}},
#                                 "embedding_norm": {"$sqrt": {"$reduce": {
#                                     "input": "$embedding",
#                                     "initialValue": 0,
#                                     "in": {"$add": ["$$value", {"$pow": ["$$this", 2]}]}
#                                 }}}
#                             },
#                             "in": {
#                                 "$divide": [
#                                     "$$dot_product",
#                                     {"$multiply": ["$$query_norm", "$$embedding_norm"]}
#                                 ]
#                             }
#                         }
#                     }
#                 }
#             },
#             {"$sort": {"similarity": -1}},
#             {"$limit": top_k},
#             {
#                 "$lookup": {
#                     "from": "documents",
#                     "localField": "doc_id",
#                     "foreignField": "_id",
#                     "as": "document"
#                 }
#             },
#             {"$unwind": "$document"},
#             {
#                 "$project": {
#                     "doc_id": 1,
#                     "similarity": 1,
#                     "filename": "$document.filename",
#                     "content": "$document.text_content"
#                 }
#             }
#         ]
        
#         results = list(self.db.document_embeddings.aggregate(pipeline))
#         self.logger.info(f"Busca semântica retornou {len(results)} resultados para a query: '{query}'")
#         return results

#     def _expand_query(self, query: str) -> List[str]:
#         """Expande a consulta com termos relacionados"""
#         # Implementação básica - pode ser melhorada com um modelo de expansão de consultas
#         expansions = [query]
        
#         # Adicionar sinônimos para termos comuns
#         synonyms = {
#             "matrícula": ["inscrição", "registro"],
#             "calificaciones": ["notas", "puntuaciones"],
#             "horario": ["cronograma", "programação"]
#         }
        
#         for term, syns in synonyms.items():
#             if term.lower() in query.lower():
#                 for syn in syns:
#                     expanded = query.lower().replace(term.lower(), syn)
#                     expansions.append(expanded)
        
#         self.logger.info(f"Consulta expandida: {expansions}")
#         return expansions

#     def _get_text_similarity(self, text1: str, text2: str) -> float:
#         """Calcula a similaridade entre dois textos"""
#         if not text1 or not text2:
#             return 0.0
            
#         # Embeddings de sentenças
#         emb1 = self.sentence_model.encode(text1)
#         emb2 = self.sentence_model.encode(text2)
        
#         similarity = cosine_similarity([emb1], [emb2])[0][0]
#         self.logger.debug(f"Similaridade entre textos: '{text1[:50]}...' e '{text2[:50]}...' = {similarity:.2f}")
#         return similarity

#     def _get_context_around(self, text: str, target: str, window: int = 5) -> str:
#         """Obtém o contexto ao redor do fragmento relevante"""
#         sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
#         try:
#             idx = next(i for i, s in enumerate(sentences) if target in s)
#             start = max(0, idx - window)
#             end = min(len(sentences), idx + window + 1)
#             context = ' '.join(sentences[start:end])
#             self.logger.debug(f"Contexto encontrado: {context[:200]}...")
#             return context
#         except (ValueError, StopIteration):
#             self.logger.debug("Fragmento alvo não encontrado no texto, retornando o alvo original")
#             return target

#     def _find_most_relevant(self, content: str, query: str) -> Dict:
#         """Encontra o fragmento mais relevante com contexto"""
#         sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', content) if s.strip()]
#         if not sentences:
#             return {"text": content[:500], "score": 0, "exact_match": ""}
        
#         # Encontrar a sentença mais similar
#         best_sentence = max(sentences, key=lambda s: self._get_text_similarity(query, s))
#         similarity = self._get_text_similarity(query, best_sentence)
#         context = self._get_context_around(content, best_sentence, self.context_window)
        
#         self.logger.info(f"Fragmento mais relevante encontrado (score: {similarity:.2f}): {best_sentence[:100]}...")
        
#         return {
#             "text": context,
#             "score": similarity,
#             "exact_match": best_sentence
#         }

#     def _format_document_response(self, content: str, source: str, query: str, score: float) -> str:
#         """Formata a resposta de um documento encontrado"""
#         relevant = self._find_most_relevant(content, query)
        
#         confidence = ""
#         if score > 0.7:
#             confidence = " (alta confiança)"
#         elif score > 0.5:
#             confidence = " (média confiança)"
#         else:
#             confidence = " (baixa confiança)"
        
#         response = (f"📌 **Informação encontrada em '{source}'{confidence}:**\n\n"
#                    f"{relevant['text']}\n\n"
#                    f"🔍 _Fragmento mais relevante:_\n"
#                    f"_{relevant['exact_match']}_\n\n"
#                    f"Gostaria de mais informações sobre este tema ou prefere buscar algo diferente?")
        
#         self.logger.info(f"Resposta formatada para documento {source} com score {score:.2f}")
#         return response

#     def _format_multiple_results(self, matches: List[Dict], query: str) -> str:
#         """Formata múltiplos resultados encontrados"""
#         if not matches:
#             return self._format_fallback_response(query)
            
#         response = "🔍 **Encontrei várias informações relevantes:**\n\n"
        
#         for i, match in enumerate(matches[:self.top_k]):
#             title = match.get('filename', 'Documento sem título')
#             score = match.get('similarity', 0)
            
#             # Obter o fragmento mais relevante
#             relevant = self._find_most_relevant(match['content'], query)
#             preview = relevant['text'][:200] + ('...' if len(relevant['text']) > 200 else '')
            
#             response += (f"{i+1}. **{title}** (similaridade: {score:.2f})\n"
#                         f"   {preview}\n\n")
        
#         response += "Por favor, indique qual destas opções te interessa mais (1, 2, 3) ou se prefere que eu busque algo diferente."
        
#         self.logger.info(f"Formatados {len(matches)} resultados múltiplos para a query")
#         return response

#     def _format_fallback_response(self, query: str) -> str:
#         """Resposta quando não encontra informação específica"""
#         # Analisar a consulta para sugestões mais inteligentes
#         query_lower = query.lower()
        
#         suggestions = []
#         if any(word in query_lower for word in ["matrícula", "inscrição", "registro"]):
#             suggestions = [
#                 "Processo de matrícula",
#                 "Requisitos de inscrição",
#                 "Datas limite de registro"
#             ]
#         elif any(word in query_lower for word in ["examen", "prova", "calificación"]):
#             suggestions = [
#                 "Calendário de exames",
#                 "Normativas de avaliação",
#                 "Processo de revisão de calificações"
#             ]
#         else:
#             suggestions = [
#                 "Regulamento acadêmico",
#                 "Serviços estudantis",
#                 "Oferta de cursos"
#             ]
        
#         response = ("Não encontrei informação exata sobre sua consulta, mas sugiro estes temas relacionados:\n\n")
#         response += "\n".join([f"• {item}" for item in suggestions])
#         response += "\n\nTe interessa algum destes ou prefere que eu busque algo diferente?"
        
#         self.logger.info(f"Retornando resposta fallback para query: '{query}'")
#         return response

#     def run(self, dispatcher, tracker, domain):
#         query = tracker.latest_message.get('text', '').strip()
#         self.logger.info(f"Processando consulta: '{query}'")
        
#         try:
#             # Primeiro tentar busca semântica com RAG
#             semantic_results = self._semantic_search(query, top_k=self.top_k)
            
#             if semantic_results:
#                 # Se houver um resultado claramente melhor, mostrá-lo
#                 if semantic_results[0]['similarity'] > 0.7:
#                     best_match = semantic_results[0]
#                     response_text = self._format_document_response(
#                         best_match['content'],
#                         best_match['filename'],
#                         query,
#                         best_match['similarity']
#                     )
#                 else:
#                     # Mostrar múltiplas opções
#                     response_text = self._format_multiple_results(semantic_results, query)
#             else:
#                 # Fallback para busca tradicional
#                 expanded_query = self._expand_query(query)
#                 all_docs = []
                
#                 for q in expanded_query:
#                     docs = list(self.collection.find(
#                         {"$text": {"$search": q}},
#                         {"score": {"$meta": "textScore"}, "text_content": 1, "filename": 1}
#                     ).limit(5))
#                     all_docs.extend(docs)
                
#                 if all_docs:
#                     # Processar documentos encontrados
#                     responses = []
#                     for doc in all_docs[:self.top_k]:
#                         content = doc.get("text_content", "")
#                         source = doc.get("filename", "documento")
                        
#                         relevant = self._find_most_relevant(content, query)
#                         if relevant['score'] >= self.min_similarity:
#                             responses.append({
#                                 'content': content,
#                                 'source': source,
#                                 'score': relevant['score'],
#                                 'similarity': relevant['score']
#                             })
                    
#                     if responses:
#                         responses.sort(key=lambda x: x['score'], reverse=True)
#                         response_text = self._format_multiple_results(responses, query)
#                     else:
#                         response_text = self._format_fallback_response(query)
#                 else:
#                     response_text = self._format_fallback_response(query)
            
#             dispatcher.utter_message(text=response_text)
                
#         except Exception as e:
#             self.logger.error(f"Erro na busca: {str(e)}", exc_info=True)
#             dispatcher.utter_message(
#                 text="Ocorreu um erro ao processar sua solicitação. Por favor, tente novamente mais tarde ou reformule sua pergunta."
#             )

#         return []
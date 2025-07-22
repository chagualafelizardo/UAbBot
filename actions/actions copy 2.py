# from typing import Dict, Text, Any, List, Optional
# from rasa_sdk import Action, Tracker
# from rasa_sdk.executor import CollectingDispatcher
# from rasa_sdk.events import SlotSet
# from pymongo import MongoClient
# import logging

# logger = logging.getLogger(__name__)

# class MongoDBManager:
#     def __init__(self):
#         self.client = MongoClient('mongodb://root:root@uabbot-mongodb-1:27017/')
#         self.db = self.client['uab']
#         self.cursos_collection = self.db['cursos']

#     def get_cursos(self, nivel: str = None) -> List[Dict]:
#         """Busca cursos por nível com informações essenciais incluindo link de detalhes"""
#         query = {}
#         if nivel:
#             query['$or'] = [
#                 {'tipo': nivel},
#                 {'nome': {'$regex': nivel, '$options': 'i'}}
#             ]
        
#         projection = {
#             'nome': 1,
#             'departamento': 1,
#             'tipo': 1,
#             'regulamentacao.regime': 1,
#             'apresentacao.url_detalhes': 1,
#             '_id': 0
#         }
        
#         try:
#             cursos = list(self.cursos_collection.find(query, projection).limit(5))
            
#             if not cursos and nivel:
#                 cursos = list(self.cursos_collection.find(
#                     {'nome': {'$regex': '|'.join(self._get_keywords_for_level(nivel)), '$options': 'i'}},
#                     projection
#                 ).limit(5))
                
#             return cursos
#         except Exception as e:
#             logger.error(f"Erro ao buscar cursos: {str(e)}")
#             return []

#     def get_curso_details(self, course_name: str) -> Optional[Dict]:
#         """Busca todos os detalhes de um curso incluindo link oficial"""
#         try:
#             curso = self.cursos_collection.find_one(
#                 {'nome': {'$regex': course_name, '$options': 'i'}},
#                 {'_id': 0}
#             )
            
#             if not curso:
#                 return None
                
#             return self._format_curso_details(curso)
#         except Exception as e:
#             logger.error(f"Erro ao buscar detalhes: {str(e)}")
#             return None

#     def _format_curso_details(self, curso: Dict) -> Dict:
#         """Formata os dados do curso para exibição"""
#         url_detalhes = curso.get('apresentacao', {}).get('url_detalhes', '#')
        
#         return {
#             'nome': curso.get('nome', ''),
#             'nivel': curso.get('tipo', ''),
#             'url_detalhes': url_detalhes,
#             'departamento': curso.get('departamento', 'Departamento não especificado'),
#             'descricao': curso.get('descricao', 'Descrição não disponível'),
#             'regime': curso.get('regulamentacao', {}).get('regime', ''),
#             'lingua': curso.get('regulamentacao', {}).get('lingua', ''),
#             'publico_alvo': self._format_list_items(curso.get('publico_alvo', [])),
#             'coordenacao': self._format_coordenacao(curso.get('coordenacao', {})),
#             'ects_total': curso.get('estrutura_curricular', {}).get('maior', {}).get('ects', 0),
#             'minors': self._format_minors(curso.get('estrutura_curricular', {}).get('minors', [])),
#             'disciplinas': self._format_disciplinas(curso.get('estrutura_curricular', {}).get('disciplinas', []))
#         }

#     def _format_list_items(self, items: List[str]) -> str:
#         return "\n".join(f"• {item}" for item in items) if items else "Não especificado"

#     def _format_coordenacao(self, coordenacao: Dict) -> str:
#         coord = coordenacao.get('coordenador', '')
#         vice = coordenacao.get('vice_coordenador', '')
#         return f"{coord} (Coordenador)\n{vice} (Vice-Coordenador)" if coord or vice else "Não especificado"

#     def _format_minors(self, minors: List[Dict]) -> str:
#         return "\n".join(
#             f"• {minor['nome']} ({minor['ects']} ECTS)"
#             for minor in minors
#         ) if minors else "Nenhum minor disponível"

#     def _format_disciplinas(self, disciplinas: List[Dict]) -> Dict[str, List]:
#         anos = {}
#         for disciplina in disciplinas:
#             ano_key = f"{disciplina['ano']}º Ano - {disciplina['semestre']}"
#             if ano_key not in anos:
#                 anos[ano_key] = []
#             anos[ano_key].append(disciplina)
#         return anos

#     def close(self):
#         self.client.close()

# # Classe para buscar cursos
# class ActionSearchUAbCourses(Action):
#     def name(self) -> Text:
#         return "action_search_uab_courses"

#     def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
#         db = None
#         try:
#             db = MongoDBManager()
            
#             # Extrair entidades diretamente do texto da mensagem
#             user_message = tracker.latest_message.get('text', '').strip()
#             course_type, course_name = self._extract_entities(user_message)

#             # Debug: Mostrar entidades detectadas
#             logger.debug(f"Entities detected - type: {course_type}, name: {course_name}")

#             # 1. Busca exata pelo nome completo do curso
#             if course_name:
#                 normalized_name = self._normalize_course_name(course_name)
#                 exact_course = db.cursos_collection.find_one(
#                     {'nome': {'$regex': f'^{normalized_name}$', '$options': 'i'}}
#                 )
                
#                 if exact_course:
#                     logger.debug(f"Exact course found: {exact_course['nome']}")
#                     return self._show_single_course(db, dispatcher, exact_course)

#             # 2. Busca por termos similares (incluindo parcial)
#             if course_name:
#                 similar_courses = list(db.cursos_collection.find(
#                     {'nome': {'$regex': course_name, '$options': 'i'}}
#                 ).limit(5))
                
#                 if similar_courses:
#                     logger.debug(f"Similar courses found: {[c['nome'] for c in similar_courses]}")
#                     if len(similar_courses) == 1:
#                         return self._show_single_course(db, dispatcher, similar_courses[0])
#                     return self._show_similar_options(dispatcher, similar_courses, course_name)

#             # 3. Busca por tipo de curso
#             if course_type:
#                 type_courses = list(db.cursos_collection.find({
#                     '$or': [
#                         {'tipo': course_type},
#                         {'nome': {'$regex': course_type, '$options': 'i'}}
#                     ]
#                 }).limit(10))
                
#                 if type_courses:
#                     return self._show_courses_list(dispatcher, type_courses, course_type)

#             # 4. Nenhum resultado encontrado
#             self._show_not_found(dispatcher, course_type, course_name)
#             return []
            
#         except Exception as e:
#             logger.error(f"Erro na busca: {str(e)}", exc_info=True)
#             dispatcher.utter_message(text="Estou com dificuldades para acessar os cursos. Tente mais tarde.")
#             return []
#         finally:
#             if db:
#                 db.close()

#     def _extract_entities(self, message: str) -> Tuple[Optional[str], Optional[str]]:
#         """Extrai entidades diretamente do texto usando lógica personalizada"""
#         # Padrões para detectar nomes completos de cursos
#         patterns = [
#             r'(licenciatura|mestrado|doutoramento)\s+(em|de)\s+([^\.\?\!]+)',
#             r'curso\s+(de|em)\s+([^\.\?\!]+)'
#         ]
        
#         course_type = None
#         course_name = None
        
#         # Verifica se é um nome completo de curso
#         for pattern in patterns:
#             match = re.search(pattern, message, re.IGNORECASE)
#             if match:
#                 if len(match.groups()) >= 3:
#                     course_type = match.group(1).lower()
#                     course_name = match.group(3).strip()
#                 else:
#                     course_name = match.group(2).strip()
#                 break
        
#         # Se não encontrou padrão completo, verifica palavras-chave
#         if not course_name:
#             # Mapeamento de tipos de curso
#             type_keywords = {
#                 'licenciatura': ['licenciatura', 'graduação', '1º ciclo'],
#                 'mestrado': ['mestrado', 'pós-graduação', '2º ciclo'],
#                 'doutoramento': ['doutoramento', 'phd', '3º ciclo']
#             }
            
#             # Verifica se mencionou tipo de curso
#             for c_type, keywords in type_keywords.items():
#                 if any(kw in message.lower() for kw in keywords):
#                     course_type = c_type
#                     break
            
#             # Extrai o nome do curso (últimas 2-4 palavras)
#             words = message.split()
#             if len(words) >= 2:
#                 course_name = ' '.join(words[-3:]) if len(words) >= 3 else ' '.join(words[-2:])
        
#         return course_type, course_name

#     def _normalize_course_name(self, name: str) -> str:
#         """Normaliza o nome do curso para busca"""
#         return name.strip().title()

#     def _show_single_course(self, db, dispatcher: CollectingDispatcher, course: Dict) -> List[Dict]:
#         """Mostra detalhes de um único curso"""
#         formatted = db._format_curso_details(course)
#         response = f"""
# 📚 *{formatted['nome']}* ({formatted['nivel']})

# 🏛️ Departamento: {formatted['departamento']}
# 🌐 Regime: {formatted['regime']} | Língua: {formatted['lingua']}
# 📝 ECTS: {formatted['ects_total']}

# 📖 Descrição:
# {formatted['descricao'][:200]}... [ler mais]({formatted['url_detalhes']})

# 🔗 [Acessar página oficial]({formatted['url_detalhes']})
# """
#         buttons = [
#             {
#                 "title": "📚 Ver disciplinas",
#                 "payload": f'/ask_course_subjects{{"course_name":"{course["nome"]}"}}'
#             },
#             {
#                 "title": "🌐 Página oficial",
#                 "url": formatted['url_detalhes'],
#                 "payload": "/external_link"
#             }
#         ]
#         dispatcher.utter_message(text=response, buttons=buttons)
#         return [
#             SlotSet("course_name", course['nome']),
#             SlotSet("course_url", formatted['url_detalhes'])
#         ]

#     def _show_similar_options(self, dispatcher: CollectingDispatcher, courses: List[Dict], original_query: str) -> List[Dict]:
#         """Mostra cursos similares quando não encontra exato"""
#         buttons = [{
#             "title": course['nome'],
#             "payload": f'/get_course_details{{"course_name":"{course["nome"]}"}}'
#         } for course in courses]
        
#         dispatcher.utter_message(
#             text=f"Encontrei estes cursos relacionados com '{original_query}':",
#             buttons=buttons
#         )
#         return []

#     def _show_courses_list(self, dispatcher: CollectingDispatcher, courses: List[Dict], course_type: str) -> List[Dict]:
#         """Lista cursos quando a busca é por tipo"""
#         courses_list = "\n".join([f"• {course['nome']}" for course in courses])
#         dispatcher.utter_message(text=f"Cursos de {course_type.lower()}:\n{courses_list}")
        
#         buttons = [{
#             "title": f"Ver {course['nome']}",
#             "payload": f'/get_course_details{{"course_name":"{course["nome"]}"}}'
#         } for course in courses[:3]]
        
#         dispatcher.utter_message(text="Selecione para ver detalhes:", buttons=buttons)
#         return [SlotSet("course_type", course_type)]

#     def _show_not_found(self, dispatcher: CollectingDispatcher, course_type: Optional[str], course_name: Optional[str]):
#         """Mensagem quando não encontra resultados"""
#         message = "Não encontrei "
#         if course_name:
#             message += f"o curso '{course_name}'"
#             if course_type:
#                 message += f" do tipo {course_type}"
#         elif course_type:
#             message += f"cursos de {course_type}"
#         else:
#             message += "nenhum curso"
        
#         dispatcher.utter_message(
#             text=message + ".",
#             buttons=[{"title": "Ver todos os cursos", "payload": "/ask_uab_courses"}]
#         )
    
# # Classe para buscar os detalhes dos cursos
# class ActionGetCourseDetails(Action):
#     def name(self) -> Text:
#         return "action_get_course_details"

#     def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
#         db = None
#         try:
#             course_name = tracker.get_slot("course_name")
#             if not course_name:
#                 dispatcher.utter_message(text="Qual curso deseja ver?")
#                 return []

#             db = MongoDBManager()
#             course = db.get_curso_details(course_name)

#             if not course:
#                 dispatcher.utter_message(
#                     text=f"Curso '{course_name}' não encontrado.",
#                     buttons=[{"title": "Ver cursos", "payload": "/ask_uab_courses"}]
#                 )
#                 return []

#             response = self._format_course_response(course)
#             buttons = self._create_detail_buttons(course)
            
#             dispatcher.utter_message(text=response, buttons=buttons)
#             return [SlotSet("course_name", course['nome']), SlotSet("course_url", course['url_detalhes'])]

#         except Exception as e:
#             logger.error(f"Erro ao buscar detalhes: {str(e)}", exc_info=True)
#             dispatcher.utter_message(text="Não consegui acessar os detalhes deste curso.")
#             return []
#         finally:
#             if db:
#                 db.close()

#     def _format_course_response(self, course: Dict) -> str:
#         return f"""
# 📚 *{course['nome']}* ({course['nivel'].capitalize()})

# 🏛️ *Departamento:* {course['departamento']}
# 🌐 *Regime:* {course['regime']} | *Língua:* {course['lingua']}
# 📝 *ECTS Totais:* {course['ects_total']}

# 📖 *Descrição:*
# {course['descricao']}

# 👥 *Público-Alvo:*
# {course['publico_alvo']}

# 👨‍🏫 *Coordenação:*
# {course['coordenacao']}

# 🎓 *Minors Disponíveis:*
# {course['minors']}

# 🔗 [Detalhes completos do curso]({course['url_detalhes']})
# """

#     def _create_detail_buttons(self, course: Dict) -> List[Dict]:
#         return [
#             {
#                 "title": "Ver disciplinas",
#                 "payload": f"/ask_course_subjects{{\"course_name\":\"{course['nome']}\"}}"
#             },
#             {
#                 "title": "Acessar página oficial",
#                 "url": course['url_detalhes'],
#                 "payload": "/external_link"
#             },
#             {
#                 "title": "Mais cursos",
#                 "payload": "/ask_uab_courses"
#             }
#         ]

# # Classe para buscar os cursos
# class ActionGetCourseSubjects(Action):
#     def name(self) -> Text:
#         return "action_get_course_subjects"

#     def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
#         db = None
#         try:
#             course_name = tracker.get_slot("course_name")
#             if not course_name:
#                 dispatcher.utter_message(text="De qual curso quer ver as disciplinas?")
#                 return []

#             db = MongoDBManager()
#             curso = db.cursos_collection.find_one(
#                 {'nome': {'$regex': course_name, '$options': 'i'}},
#                 {'estrutura_curricular.disciplinas': 1, 'nome': 1, '_id': 0}
#             )

#             if not curso or not curso.get('estrutura_curricular', {}).get('disciplinas'):
#                 dispatcher.utter_message(
#                     text=f"Não encontrei disciplinas para '{course_name}'.",
#                     buttons=[{
#                         "title": "Voltar ao curso",
#                         "payload": f'/get_course_details{{"course_name":"{course_name}"}}'
#                     }]
#                 )
#                 return []

#             response = self._format_subjects_response(curso)
#             dispatcher.utter_message(text=response, buttons=self._create_subject_buttons(course_name))
            
#             return []
        
#         except Exception as e:
#             logger.error(f"Erro ao buscar disciplinas: {str(e)}", exc_info=True)
#             dispatcher.utter_message(text="Não consegui acessar as disciplinas.")
#             return []
#         finally:
#             if db:
#                 db.close()

#     def _format_subjects_response(self, curso: Dict) -> str:
#         disciplinas = curso['estrutura_curricular']['disciplinas']
#         anos = {}
        
#         for disciplina in disciplinas:
#             ano_key = f"{disciplina['ano']}º Ano - {disciplina['semestre']}"
#             if ano_key not in anos:
#                 anos[ano_key] = []
#             anos[ano_key].append(disciplina)

#         response = f"📚 *Disciplinas de {curso['nome']}*\n\n"
#         for ano, disciplinas_ano in anos.items():
#             response += f"*{ano}:*\n"
#             for disciplina in disciplinas_ano[:5]:
#                 response += f"• {disciplina['codigo']} - {disciplina['nome']} ({disciplina['ects']} ECTS)\n"
#             response += "\n"

#         if len(disciplinas) > 10:
#             response += f"ℹ️ Mostrando 10 de {len(disciplinas)} disciplinas."
        
#         return response

#     def _create_subject_buttons(self, course_name: str) -> List[Dict]:
#         return [
#             {
#                 "title": "Voltar ao curso",
#                 "payload": f'/get_course_details{{"course_name":"{course_name}"}}'
#             },
#             {
#                 "title": "Outros cursos",
#                 "payload": "/ask_uab_courses"
#             }
#         ]

# # Classe para buscar as Frequents Quantions And Answer  
# class ActionSearchFAQ(Action):
#     def name(self) -> Text:
#         return "action_search_faq"
    
#     async def run(self, dispatcher: CollectingDispatcher,
#                 tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
#         # Primeiro verificamos se realmente é uma FAQ
#         if not self._is_faq_question(tracker):
#             logger.debug("Mensagem não identificada como FAQ - reclassificando")
#             return await self._handle_non_faq(dispatcher, tracker)
            
#         try:
#             client = MongoClient('mongodb://root:root@uabbot-mongodb-1:27017/')
#             db = client['uab']
#             faqs = db['faqs']
            
#             user_message = tracker.latest_message.get('text')
#             topic = next(tracker.get_latest_entity_values("faq_topic"), None)

#             # Busca otimizada com fallback
#             results = await self._search_faqs_with_fallback(faqs, user_message, topic)
            
#             if results:
#                 return self._format_faq_response(dispatcher, results)
                
#             return self._handle_faq_not_found(dispatcher, user_message)
            
#         except Exception as e:
#             logger.error(f"Erro ao buscar FAQ: {str(e)}")
#             return self._handle_search_error(dispatcher)

#     def _is_faq_question(self, tracker: Tracker) -> bool:
#         """Determina se a mensagem é realmente uma pergunta de FAQ"""
#         message = tracker.latest_message.get('text', '').lower()
#         intent = tracker.latest_message.get('intent', {}).get('name')
        
#         # Palavras-chave que indicam claramente uma FAQ
#         faq_keywords = [
#             'como', 'quando', 'onde', 'quem', 'qual', 
#             'quais', 'posso', 'devo', 'preciso', 'dúvida',
#             'candidatar', 'processo', 'documento', 'prazo',
#             'requisito', 'informação', 'ajuda','admissao',
#             'Admissão', 'Apoio Académico', 'Publicações',
#             'Científicas', 'Publicações Científicas',
#             ''
#         ]
        
#         # Verifica se há entidade de FAQ ou palavras-chave
#         has_faq_entity = any(e['entity'] == 'faq_topic' 
#                            for e in tracker.latest_message.get('entities', []))
        
#         is_faq_intent = intent == 'ask_faq'
#         contains_keywords = any(kw in message for kw in faq_keywords)
        
#         return has_faq_entity or is_faq_intent or contains_keywords

#     async def _handle_non_faq(self, dispatcher: CollectingDispatcher, tracker: Tracker):
#         """Lida com mensagens que não são FAQs"""
#         message = tracker.latest_message.get('text', '')
        
#         # Verifica se parece ser sobre cursos
#         course_keywords = ['licenciatura', 'mestrado', 'doutoramento', 'curso', 'disciplina']
#         if any(kw in message.lower() for kw in course_keywords):
#             dispatcher.utter_message(text="Parece que você está perguntando sobre cursos. Vou ajudar com isso.")
#             return [ActionExecuted("action_search_uab_courses")]
        
#         # Fallback genérico
#         dispatcher.utter_message(text="Não consegui identificar sua solicitação. Você pode reformular?")
#         return []

#     async def _search_faqs_with_fallback(self, faqs_collection, user_message: str, topic: Optional[str]):
#         """Busca FAQs com estratégia de fallback"""
#         # Primeira tentativa: busca exata por tópico
#         if topic:
#             results = list(faqs_collection.find(
#                 {"topico": topic},
#                 {"pergunta": 1, "resposta": 1, "detalhes_adicionais": 1}
#             ).limit(3))
#             if results:
#                 return results

#         # Segunda tentativa: busca por similaridade textual
#         results = list(faqs_collection.find(
#             {"$text": {"$search": user_message}},
#             {"score": {"$meta": "textScore"}, "pergunta": 1, "resposta": 1}
#         ).sort([("score", {"$meta": "textScore"})]).limit(3))

#         return results if results else None

#     def _format_faq_response(self, dispatcher: CollectingDispatcher, results: List[Dict]) -> List[Dict]:
#         """Formata a resposta da FAQ"""
#         best = results[0]
#         response = best["resposta"]

#         if len(results) > 1:
#             related_topics = {f["topico"] for f in results[1:] if f.get("topico")}
#             if related_topics:
#                 buttons = [{
#                     "title": t,
#                     "payload": f'/ask_faq{{"faq_topic":"{t}"}}'
#                 } for t in related_topics]
#                 dispatcher.utter_message(text=response, buttons=buttons)
#                 return []

#         dispatcher.utter_message(text=response)
#         return []

#     def _handle_faq_not_found(self, dispatcher: CollectingDispatcher, user_message: str) -> List[Dict]:
#         """Lida com casos onde a FAQ não foi encontrada"""
#         logger.warning(f"FAQ não encontrada para: {user_message}")
#         dispatcher.utter_message(response="utter_faq_not_found")
#         return []

#     def _handle_search_error(self, dispatcher: CollectingDispatcher) -> List[Dict]:
#         """Lida com erros na busca"""
#         dispatcher.utter_message(text="Ocorreu um erro ao acessar nossa base de conhecimento.")
#         return []

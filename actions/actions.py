from typing import Dict, Text, Any, List, Optional
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet
from pymongo import MongoClient
import logging

logger = logging.getLogger(__name__)

class MongoDBManager:
    def __init__(self):
        self.client = MongoClient('mongodb://root:root@uabbot-mongodb-1:27017/')
        self.db = self.client['uab']
        self.cursos_collection = self.db['cursos']

    def get_cursos(self, nivel: str = None) -> List[Dict]:
        """Busca cursos por nível com informações essenciais incluindo link de detalhes"""
        query = {}
        if nivel:
            query['$or'] = [
                {'tipo': nivel},
                {'nome': {'$regex': nivel, '$options': 'i'}}
            ]
        
        projection = {
            'nome': 1,
            'departamento': 1,
            'tipo': 1,
            'regulamentacao.regime': 1,
            'apresentacao.url_detalhes': 1,
            '_id': 0
        }
        
        try:
            cursos = list(self.cursos_collection.find(query, projection).limit(5))
            
            if not cursos and nivel:
                cursos = list(self.cursos_collection.find(
                    {'nome': {'$regex': '|'.join(self._get_keywords_for_level(nivel)), '$options': 'i'}},
                    projection
                ).limit(5))
                
            return cursos
        except Exception as e:
            logger.error(f"Erro ao buscar cursos: {str(e)}")
            return []

    def get_curso_details(self, course_name: str) -> Optional[Dict]:
        """Busca todos os detalhes de um curso incluindo link oficial"""
        try:
            curso = self.cursos_collection.find_one(
                {'nome': {'$regex': course_name, '$options': 'i'}},
                {'_id': 0}
            )
            
            if not curso:
                return None
                
            return self._format_curso_details(curso)
        except Exception as e:
            logger.error(f"Erro ao buscar detalhes: {str(e)}")
            return None

    def _format_curso_details(self, curso: Dict) -> Dict:
        """Formata os dados do curso para exibição"""
        url_detalhes = curso.get('apresentacao', {}).get('url_detalhes', '#')
        
        return {
            'nome': curso.get('nome', ''),
            'nivel': curso.get('tipo', ''),
            'url_detalhes': url_detalhes,
            'departamento': curso.get('departamento', 'Departamento não especificado'),
            'descricao': curso.get('descricao', 'Descrição não disponível'),
            'regime': curso.get('regulamentacao', {}).get('regime', ''),
            'lingua': curso.get('regulamentacao', {}).get('lingua', ''),
            'publico_alvo': self._format_list_items(curso.get('publico_alvo', [])),
            'coordenacao': self._format_coordenacao(curso.get('coordenacao', {})),
            'ects_total': curso.get('estrutura_curricular', {}).get('maior', {}).get('ects', 0),
            'minors': self._format_minors(curso.get('estrutura_curricular', {}).get('minors', [])),
            'disciplinas': self._format_disciplinas(curso.get('estrutura_curricular', {}).get('disciplinas', []))
        }

    def _format_list_items(self, items: List[str]) -> str:
        return "\n".join(f"• {item}" for item in items) if items else "Não especificado"

    def _format_coordenacao(self, coordenacao: Dict) -> str:
        coord = coordenacao.get('coordenador', '')
        vice = coordenacao.get('vice_coordenador', '')
        return f"{coord} (Coordenador)\n{vice} (Vice-Coordenador)" if coord or vice else "Não especificado"

    def _format_minors(self, minors: List[Dict]) -> str:
        return "\n".join(
            f"• {minor['nome']} ({minor['ects']} ECTS)"
            for minor in minors
        ) if minors else "Nenhum minor disponível"

    def _format_disciplinas(self, disciplinas: List[Dict]) -> Dict[str, List]:
        anos = {}
        for disciplina in disciplinas:
            ano_key = f"{disciplina['ano']}º Ano - {disciplina['semestre']}"
            if ano_key not in anos:
                anos[ano_key] = []
            anos[ano_key].append(disciplina)
        return anos

    def close(self):
        self.client.close()

# Classe para buscar cursos
class ActionSearchUAbCourses(Action):
    def name(self) -> Text:
        return "action_search_uab_courses"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        db = None
        try:
            db = MongoDBManager()
            
            # Extrair entidades da mensagem do usuário
            course_type = next(tracker.get_latest_entity_values("course_type"), None)
            course_name = next(tracker.get_latest_entity_values("course_name"), None)
            
            # Se não encontrar entidades, tentar detectar pelo texto
            if not course_type:
                course_type = self._detect_course_type(tracker.latest_message.get('text', ''))
            
            query = {}
            
            # Construir query baseada no que foi fornecido
            if course_type and course_name:
                query = {
                    '$and': [
                        {'$or': [
                            {'tipo': course_type},
                            {'nome': {'$regex': course_type, '$options': 'i'}}
                        ]},
                        {'nome': {'$regex': course_name, '$options': 'i'}}
                    ]
                }
            elif course_type:
                query = {
                    '$or': [
                        {'tipo': course_type},
                        {'nome': {'$regex': course_type, '$options': 'i'}}
                    ]
                }
            elif course_name:
                query = {'nome': {'$regex': course_name, '$options': 'i'}}
            else:
                dispatcher.utter_message(response="utter_ask_course_type")
                return []

            # Buscar cursos com a query construída
            courses = list(db.cursos_collection.find(query).limit(10))
            
            if not courses:
                # Tentar busca mais ampla se não encontrar resultados
                if course_name:
                    courses = list(db.cursos_collection.find(
                        {'nome': {'$regex': course_name, '$options': 'i'}}
                    ).limit(5))
                
                if not courses:
                    message = "Não encontrei cursos"
                    if course_type:
                        message += f" de {course_type}"
                    if course_name:
                        message += f" com o nome '{course_name}'"
                    
                    dispatcher.utter_message(
                        text=message + " disponíveis.",
                        buttons=[{"title": "Ver todos os cursos", "payload": "/ask_uab_courses"}]
                    )
                    return []

            # Se encontrou apenas um curso, mostrar diretamente os detalhes
            if len(courses) == 1:
                course = courses[0]
                formatted = db._format_curso_details(course)
                response = self._format_course_response(formatted)
                buttons = self._create_detail_buttons(formatted)
                dispatcher.utter_message(text=response, buttons=buttons)
                return [SlotSet("course_name", course['nome']), SlotSet("course_url", formatted['url_detalhes'])]
            
            # Para múltiplos cursos, mostrar lista resumida
            dispatcher.utter_message(
                text=f"🔍 Encontrei {len(courses)} cursos que correspondem à sua pesquisa:"
            )
            
            for course in courses[:5]:  # Limitar a 5 cursos
                response = self._format_course_card(course)
                buttons = self._create_course_buttons(course)
                dispatcher.utter_message(text=response, buttons=buttons)
            
            if len(courses) > 5:
                dispatcher.utter_message(
                    text=f"ℹ️ Mostrando 5 de {len(courses)} cursos encontrados. " +
                    "Você pode refinar sua pesquisa especificando melhor o que procura."
                )
            
            return [SlotSet("course_type", course_type) if course_type else None,
                    SlotSet("course_name", course_name) if course_name else None]
        
        except Exception as e:
            logger.error(f"Erro na busca: {str(e)}", exc_info=True)
            dispatcher.utter_message(text="Estou com dificuldades para acessar os cursos. Tente mais tarde.")
            return []
        finally:
            if db:
                db.close()

    def _format_course_card(self, course: Dict) -> str:
        """Formata um cartão completo com informações do curso"""
        nivel = course.get('tipo', 'Nível não especificado')
        departamento = course.get('departamento', 'Departamento não especificado')
        regime = course.get('regulamentacao', {}).get('regime', 'Regime não especificado')
        
        ects = ""
        if 'estrutura_curricular' in course and 'maior' in course['estrutura_curricular']:
            ects = f"📝 ECTS: {course['estrutura_curricular']['maior'].get('ects', 'Não especificado')}"
        
        modalidade = ""
        if 'modalidade' in course and 'ensino' in course['modalidade']:
            modalidade = f"💻 Modalidade: {course['modalidade']['ensino']}"
            if 'observacoes' in course['modalidade']:
                modalidade += f" ({course['modalidade']['observacoes']})"
        
        descricao = course.get('descricao', 'Descrição não disponível')
        if len(descricao) > 150:
            descricao = descricao[:150] + "..."
        
        url_detalhes = course.get('apresentacao', {}).get('url_detalhes', '#')
        
        return f"""
📚 *{course['nome']}* ({nivel.capitalize()})

🏛️ Departamento: {departamento}
🌐 Regime: {regime}
{ects}
{modalidade}

📖 {descricao}
🔗 [Mais detalhes]({url_detalhes})
"""

    def _create_course_buttons(self, course: Dict) -> List[Dict]:
        buttons = [
            {
                "title": "📋 Ver detalhes completos",
                "payload": f'/get_course_details{{"course_name":"{course["nome"]}"}}'
            }
        ]
        
        if 'apresentacao' in course and 'url_detalhes' in course['apresentacao']:
            buttons.append({
                "title": "🌐 Página oficial",
                "url": course['apresentacao']['url_detalhes'],
                "payload": "/external_link"
            })
            
        return buttons

    def _format_course_response(self, course: Dict) -> str:
        """Formata a resposta detalhada do curso (usada quando há apenas um resultado)"""
        return f"""
📚 *{course['nome']}* ({course['nivel'].capitalize()})

🏛️ *Departamento:* {course['departamento']}
🌐 *Regime:* {course['regime']} | *Língua:* {course['lingua']}
📝 *ECTS Totais:* {course['ects_total']}

📖 *Descrição:*
{course['descricao']}

👥 *Público-Alvo:*
{course['publico_alvo']}

👨‍🏫 *Coordenação:*
{course['coordenacao']}

🎓 *Minors Disponíveis:*
{course['minors']}

🔗 [Detalhes completos do curso]({course['url_detalhes']})
"""

    def _create_detail_buttons(self, course: Dict) -> List[Dict]:
        return [
            {
                "title": "📚 Ver disciplinas",
                "payload": f'/ask_course_subjects{{"course_name":"{course["nome"]}"}}'
            },
            {
                "title": "🌐 Acessar página oficial",
                "url": course['url_detalhes'],
                "payload": "/external_link"
            },
            {
                "title": "🔍 Mais cursos",
                "payload": "/ask_uab_courses"
            }
        ]

    def _detect_course_type(self, message: str) -> Optional[str]:
        message = message.lower()
        mapping = {
            'licenciatura': ['licenciatura', 'graduação', '1º ciclo'],
            'mestrado': ['mestrado', 'pós-graduação', '2º ciclo'],
            'doutoramento': ['doutoramento', 'phd', '3º ciclo']
        }
        for course_type, keywords in mapping.items():
            if any(keyword in message for keyword in keywords):
                return course_type
        return None


# Classe para buscar os detalhes dos cursos
class ActionGetCourseDetails(Action):
    def name(self) -> Text:
        return "action_get_course_details"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        db = None
        try:
            course_name = tracker.get_slot("course_name")
            if not course_name:
                dispatcher.utter_message(text="Qual curso deseja ver?")
                return []

            db = MongoDBManager()
            course = db.get_curso_details(course_name)

            if not course:
                dispatcher.utter_message(
                    text=f"Curso '{course_name}' não encontrado.",
                    buttons=[{"title": "Ver cursos", "payload": "/ask_uab_courses"}]
                )
                return []

            response = self._format_course_response(course)
            buttons = self._create_detail_buttons(course)
            
            dispatcher.utter_message(text=response, buttons=buttons)
            return [SlotSet("course_name", course['nome']), SlotSet("course_url", course['url_detalhes'])]

        except Exception as e:
            logger.error(f"Erro ao buscar detalhes: {str(e)}", exc_info=True)
            dispatcher.utter_message(text="Não consegui acessar os detalhes deste curso.")
            return []
        finally:
            if db:
                db.close()

    def _format_course_response(self, course: Dict) -> str:
        return f"""
📚 *{course['nome']}* ({course['nivel'].capitalize()})

🏛️ *Departamento:* {course['departamento']}
🌐 *Regime:* {course['regime']} | *Língua:* {course['lingua']}
📝 *ECTS Totais:* {course['ects_total']}

📖 *Descrição:*
{course['descricao']}

👥 *Público-Alvo:*
{course['publico_alvo']}

👨‍🏫 *Coordenação:*
{course['coordenacao']}

🎓 *Minors Disponíveis:*
{course['minors']}

🔗 [Detalhes completos do curso]({course['url_detalhes']})
"""

    def _create_detail_buttons(self, course: Dict) -> List[Dict]:
        return [
            {
                "title": "Ver disciplinas",
                "payload": f"/ask_course_subjects{{\"course_name\":\"{course['nome']}\"}}"
            },
            {
                "title": "Acessar página oficial",
                "url": course['url_detalhes'],
                "payload": "/external_link"
            },
            {
                "title": "Mais cursos",
                "payload": "/ask_uab_courses"
            }
        ]

# Classe para buscar os cursos
class ActionGetCourseSubjects(Action):
    def name(self) -> Text:
        return "action_get_course_subjects"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        db = None
        try:
            course_name = tracker.get_slot("course_name")
            if not course_name:
                dispatcher.utter_message(text="De qual curso quer ver as disciplinas?")
                return []

            db = MongoDBManager()
            curso = db.cursos_collection.find_one(
                {'nome': {'$regex': course_name, '$options': 'i'}},
                {'estrutura_curricular.disciplinas': 1, 'nome': 1, '_id': 0}
            )

            if not curso or not curso.get('estrutura_curricular', {}).get('disciplinas'):
                dispatcher.utter_message(
                    text=f"Não encontrei disciplinas para '{course_name}'.",
                    buttons=[{
                        "title": "Voltar ao curso",
                        "payload": f'/get_course_details{{"course_name":"{course_name}"}}'
                    }]
                )
                return []

            response = self._format_subjects_response(curso)
            dispatcher.utter_message(text=response, buttons=self._create_subject_buttons(course_name))
            
            return []
        
        except Exception as e:
            logger.error(f"Erro ao buscar disciplinas: {str(e)}", exc_info=True)
            dispatcher.utter_message(text="Não consegui acessar as disciplinas.")
            return []
        finally:
            if db:
                db.close()

    def _format_subjects_response(self, curso: Dict) -> str:
        disciplinas = curso['estrutura_curricular']['disciplinas']
        anos = {}
        
        for disciplina in disciplinas:
            ano_key = f"{disciplina['ano']}º Ano - {disciplina['semestre']}"
            if ano_key not in anos:
                anos[ano_key] = []
            anos[ano_key].append(disciplina)

        response = f"📚 *Disciplinas de {curso['nome']}*\n\n"
        for ano, disciplinas_ano in anos.items():
            response += f"*{ano}:*\n"
            for disciplina in disciplinas_ano[:5]:
                response += f"• {disciplina['codigo']} - {disciplina['nome']} ({disciplina['ects']} ECTS)\n"
            response += "\n"

        if len(disciplinas) > 10:
            response += f"ℹ️ Mostrando 10 de {len(disciplinas)} disciplinas."
        
        return response

    def _create_subject_buttons(self, course_name: str) -> List[Dict]:
        return [
            {
                "title": "Voltar ao curso",
                "payload": f'/get_course_details{{"course_name":"{course_name}"}}'
            },
            {
                "title": "Outros cursos",
                "payload": "/ask_uab_courses"
            }
        ]

# Classe para buscar as Frequents Quantions And Answer  
class ActionSearchFAQ(Action):
    def name(self) -> Text:
        return "action_search_faq"
    
    async def run(self, dispatcher: CollectingDispatcher,
                tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        # Primeiro verificamos se realmente é uma FAQ
        if not self._is_faq_question(tracker):
            logger.debug("Mensagem não identificada como FAQ - reclassificando")
            return await self._handle_non_faq(dispatcher, tracker)
            
        try:
            client = MongoClient('mongodb://root:root@uabbot-mongodb-1:27017/')
            db = client['uab']
            faqs = db['faqs']
            
            user_message = tracker.latest_message.get('text')
            topic = next(tracker.get_latest_entity_values("faq_topic"), None)

            # Busca otimizada com fallback
            results = await self._search_faqs_with_fallback(faqs, user_message, topic)
            
            if results:
                return self._format_faq_response(dispatcher, results)
                
            return self._handle_faq_not_found(dispatcher, user_message)
            
        except Exception as e:
            logger.error(f"Erro ao buscar FAQ: {str(e)}")
            return self._handle_search_error(dispatcher)

    def _is_faq_question(self, tracker: Tracker) -> bool:
        """Determina se a mensagem é realmente uma pergunta de FAQ"""
        message = tracker.latest_message.get('text', '').lower()
        intent = tracker.latest_message.get('intent', {}).get('name')
        
        # Palavras-chave que indicam claramente uma FAQ
        faq_keywords = [
            'como', 'quando', 'onde', 'quem', 'qual', 
            'quais', 'posso', 'devo', 'preciso', 'dúvida',
            'candidatar', 'processo', 'documento', 'prazo',
            'requisito', 'informação', 'ajuda','admissao',
            'Admissão', 'Apoio Académico', 'Publicações',
            'Científicas', 'Publicações Científicas',
            ''
        ]
        
        # Verifica se há entidade de FAQ ou palavras-chave
        has_faq_entity = any(e['entity'] == 'faq_topic' 
                           for e in tracker.latest_message.get('entities', []))
        
        is_faq_intent = intent == 'ask_faq'
        contains_keywords = any(kw in message for kw in faq_keywords)
        
        return has_faq_entity or is_faq_intent or contains_keywords

    async def _handle_non_faq(self, dispatcher: CollectingDispatcher, tracker: Tracker):
        """Lida com mensagens que não são FAQs"""
        message = tracker.latest_message.get('text', '')
        
        # Verifica se parece ser sobre cursos
        course_keywords = ['licenciatura', 'mestrado', 'doutoramento', 'curso', 'disciplina']
        if any(kw in message.lower() for kw in course_keywords):
            dispatcher.utter_message(text="Parece que você está perguntando sobre cursos. Vou ajudar com isso.")
            return [ActionExecuted("action_search_uab_courses")]
        
        # Fallback genérico
        dispatcher.utter_message(text="Não consegui identificar sua solicitação. Você pode reformular?")
        return []

    async def _search_faqs_with_fallback(self, faqs_collection, user_message: str, topic: Optional[str]):
        """Busca FAQs com estratégia de fallback"""
        # Primeira tentativa: busca exata por tópico
        if topic:
            results = list(faqs_collection.find(
                {"topico": topic},
                {"pergunta": 1, "resposta": 1, "detalhes_adicionais": 1}
            ).limit(3))
            if results:
                return results

        # Segunda tentativa: busca por similaridade textual
        results = list(faqs_collection.find(
            {"$text": {"$search": user_message}},
            {"score": {"$meta": "textScore"}, "pergunta": 1, "resposta": 1}
        ).sort([("score", {"$meta": "textScore"})]).limit(3))

        return results if results else None

    def _format_faq_response(self, dispatcher: CollectingDispatcher, results: List[Dict]) -> List[Dict]:
        """Formata a resposta da FAQ"""
        best = results[0]
        response = best["resposta"]

        if len(results) > 1:
            related_topics = {f["topico"] for f in results[1:] if f.get("topico")}
            if related_topics:
                buttons = [{
                    "title": t,
                    "payload": f'/ask_faq{{"faq_topic":"{t}"}}'
                } for t in related_topics]
                dispatcher.utter_message(text=response, buttons=buttons)
                return []

        dispatcher.utter_message(text=response)
        return []

    def _handle_faq_not_found(self, dispatcher: CollectingDispatcher, user_message: str) -> List[Dict]:
        """Lida com casos onde a FAQ não foi encontrada"""
        logger.warning(f"FAQ não encontrada para: {user_message}")
        dispatcher.utter_message(response="utter_faq_not_found")
        return []

    def _handle_search_error(self, dispatcher: CollectingDispatcher) -> List[Dict]:
        """Lida com erros na busca"""
        dispatcher.utter_message(text="Ocorreu um erro ao acessar nossa base de conhecimento.")
        return []

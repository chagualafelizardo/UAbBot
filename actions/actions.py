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


class ActionSearchUAbCourses(Action):
    def name(self) -> Text:
        return "action_search_uab_courses"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        db = None
        try:
            db = MongoDBManager()
            course_type = self._detect_course_type(tracker.latest_message.get('text', ''))
            
            if not course_type:
                dispatcher.utter_message(response="utter_ask_course_type")
                return []

            courses = db.get_cursos(nivel=course_type)
            
            if not courses:
                dispatcher.utter_message(
                    text=f"Não encontrei cursos de {course_type} disponíveis.",
                    buttons=[{"title": "Ver todos os cursos", "payload": "/ask_uab_courses"}]
                )
                return []
            
            response = self._format_courses_response(course_type, courses)
            dispatcher.utter_message(text=response, buttons=self._create_course_buttons(courses))
            
            return [SlotSet("course_type", course_type)]
        
        except Exception as e:
            logger.error(f"Erro na busca: {str(e)}", exc_info=True)
            dispatcher.utter_message(text="Estou com dificuldades para acessar os cursos. Tente mais tarde.")
            return []
        finally:
            if db:
                db.close()

    def _format_courses_response(self, course_type: str, courses: List[Dict]) -> str:
        courses_text = "\n".join(
            f"• {course['nome']} ({course['departamento']}) - {course['regulamentacao']['regime']}"
            for course in courses
        )
        return f"🔍 Cursos de {course_type} encontrados:\n\n{courses_text}"

    def _create_course_buttons(self, courses: List[Dict]) -> List[Dict]:
        return [{
            "title": f"Detalhes de {course['nome']}",
            "payload": f'/get_course_details{{"course_name":"{course["nome"]}"}}'
        } for course in courses[:3]]

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
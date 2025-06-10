from typing import Dict, Text, Any, List, Optional
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet
from .db_manager import DatabaseManager
import logging

logger = logging.getLogger(__name__)

class ActionSearchUAbCourses(Action):
    def name(self) -> Text:
        return "action_search_uab_courses"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:
        
        db = None
        try:
            db = DatabaseManager()
            message = tracker.latest_message.get('text', '').lower()
            
            course_type = tracker.get_slot("course_type") or self._detect_course_type(message)
            
            if not course_type:
                dispatcher.utter_message(response="utter_ask_course_type")
                return []

            logger.info(f"Buscando cursos do tipo: {course_type}")
            
            # Consulta simplificada para debug
            courses = db.get_cursos(nivel=course_type)
            
            if not courses:
                dispatcher.utter_message(text=f"Nenhum curso de {course_type} encontrado.")
                return [SlotSet("course_type", None)]
            
            # Resposta simplificada
            courses_list = "\n".join([f"- {c['nome']}" for c in courses[:3]])
            dispatcher.utter_message(text=f"Cursos de {course_type}:\n{courses_list}")
            
            return [SlotSet("course_type", course_type)]

        except Exception as e:
            logger.error(f"ERRO CRÍTICO: {str(e)}", exc_info=True)
            dispatcher.utter_message(text="Desculpe, estou com problemas para acessar o banco de dados.")
            return []
        finally:
            if db:
                db.close()

    def _detect_course_type(self, message: str) -> Optional[str]:
        """Detecta o tipo de curso com inteligência contextual"""
        course_types = {
            'licenciatura': ['licenciatura', 'licenciaturas', 'licença', '1° ciclo'],
            'mestrado': ['mestrado', 'mestrados', '2° ciclo', 'pós-graduação'],
            'doutoramento': ['doutoramento', 'doutoramentos', '3° ciclo', 'doutorado']
        }
        
        for course_type, keywords in course_types.items():
            if any(keyword in message for keyword in keywords):
                return course_type
        return None

class ActionGetCourseDetails(Action):
    def name(self) -> Text:
        return "action_get_course_details"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        try:
            db = DatabaseManager()
            course_name = (tracker.get_slot("course_name") or 
                          next(tracker.get_latest_entity_values("course_name"), None))

            if not course_name:
                dispatcher.utter_message(text="Por favor, especifique qual curso deseja ver.")
                return []

            logger.info(f"Buscando detalhes para: {course_name}")
            
            course = db.get_curso_details(course_name)
            
            if not course:
                dispatcher.utter_message(text=f"Curso '{course_name}' não encontrado na base de dados.")
                return []
            
            # Resposta formatada com rich content
            response = {
                "text": (
                    f"📘 *{course['nome']}* ({course['nivel']})\n\n"
                    f"📝 {course.get('descricao', 'Descrição não disponível')}\n\n"
                    f"🌐 [Acessar página do curso]({course['url']})"
                ),
                "buttons": [
                    {
                        "title": "Ver mais cursos",
                        "payload": "/search_courses"
                    }
                ]
            }
            
            dispatcher.utter_message(json_message=response)
            
            return [
                SlotSet("course_name", course['nome']),
                SlotSet("course_url", course['url'])
            ]

        except Exception as e:
            logger.error(f"Erro ao buscar detalhes: {str(e)}")
            dispatcher.utter_message(text="Ocorreu um erro ao buscar os detalhes do curso.")
            return []
        finally:
            db.close()
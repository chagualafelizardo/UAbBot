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
            
            # Detecção melhorada do tipo de curso
            course_type = self._detect_course_type(message)
            if not course_type:
                dispatcher.utter_message(response="utter_ask_course_type")
                return []

            logger.info(f"Buscando cursos do tipo: {course_type}")
            
            # Consulta ao banco de dados
            courses = db.get_cursos(nivel=course_type)
            
            if not courses:
                dispatcher.utter_message(
                    text=f"Não encontrei cursos de {course_type} disponíveis.",
                    buttons=[{
                        "title": "Ver todos os cursos",
                        "payload": "/ask_uab_courses"
                    }]
                )
                return []

            # Formatação da resposta
            courses_text = "\n".join(
                f"{idx+1}. {course['nome']}" 
                for idx, course in enumerate(courses[:5])
            )
            
            dispatcher.utter_message(
                text=f"🔍 Encontrei estes cursos de {course_type}:\n\n{courses_text}",
                buttons=[{
                    "title": f"Detalhes de {course['nome']}",
                    "payload": f'/get_course_details{{"course_name":"{course["nome"]}"}}'
                } for course in courses[:3]]
            )
            
            return [SlotSet("course_type", course_type)]

        except Exception as e:
            logger.error(f"Erro na busca de cursos: {str(e)}", exc_info=True)
            dispatcher.utter_message(
                text="Estou com dificuldades para acessar os cursos no momento. Por favor, tente mais tarde."
            )
            return []
        finally:
            if db:
                db.close()

    def _detect_course_type(self, message: str) -> Optional[str]:
        """Detecção robusta do tipo de curso"""
        mapping = {
            'licenciatura': ['licenciatura', 'licenciaturas', 'graduação', '1° ciclo', 'primeiro ciclo'],
            'mestrado': ['mestrado', 'mestrados', 'pós-graduação', '2° ciclo', 'segundo ciclo'],
            'doutoramento': ['doutoramento', 'doutoramentos', '3° ciclo', 'terceiro ciclo', 'phd']
        }
        
        for course_type, keywords in mapping.items():
            if any(keyword in message.lower() for keyword in keywords):
                return course_type
        return None

class ActionGetCourseDetails(Action):
    def name(self) -> Text:
        return "action_get_course_details"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        db = None
        try:
            db = DatabaseManager()
            course_name = tracker.get_slot("course_name") or \
                         next(tracker.get_latest_entity_values("course_name"), None)

            if not course_name:
                dispatcher.utter_message(text="Qual curso você gostaria de conhecer?")
                return []

            logger.info(f"Buscando detalhes para: {course_name}")
            
            course = db.get_curso_details(course_name)
            
            if not course:
                dispatcher.utter_message(
                    text=f"Não encontrei informações sobre '{course_name}'.",
                    buttons=[{
                        "title": "Ver lista de cursos",
                        "payload": "/ask_uab_courses"
                    }]
                )
                return []

            # Formatação rica da resposta
            response = f"""
            📚 *{course['nome']}* ({course['nivel']})
            
            {course.get('descricao', 'Descrição não disponível')}
            
            🔗 [Acessar página oficial]({course['url']})
            """
            
            dispatcher.utter_message(
                text=response,
                buttons=[
                    {
                        "title": "Mais cursos",
                        "payload": "/ask_uab_courses"
                    },
                    {
                        "title": "Fale com um atendente",
                        "payload": "/request_human"
                    }
                ]
            )
            
            return [
                SlotSet("course_name", course['nome']),
                SlotSet("course_url", course['url'])
            ]

        except Exception as e:
            logger.error(f"Erro ao buscar detalhes: {str(e)}", exc_info=True)
            dispatcher.utter_message(
                text="Não consegui acessar os detalhes deste curso no momento."
            )
            return []
        finally:
            if db:
                db.close()
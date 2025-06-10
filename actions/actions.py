from typing import Dict, Text, Any, List, Optional
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet
from .db_manager import DatabaseManager

class ActionSearchUAbCourses(Action):
    def name(self) -> Text:
        return "action_search_uab_courses"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        db = DatabaseManager()
        message = tracker.latest_message.get('text', '').lower()
        course_type = self._detect_course_type(message)
        
        # Log da conversa
        db.log_conversa(
            sender_id=tracker.sender_id,
            mensagem=message,
            intencao=tracker.latest_message.get('intent', {}).get('name'),
            resposta=None,
            contexto=tracker.current_state()
        )

        if not course_type:
            dispatcher.utter_message(response="utter_ask_course_type")
            return []

        courses = db.get_cursos(nivel=course_type.capitalize())
        db.close()

        if not courses:
            dispatcher.utter_message(text=f"Não encontrei cursos de {course_type}.")
            return [SlotSet("course_type", None)]
        
        # Mostra os cursos com botões de detalhes
        for i, course in enumerate(courses[:5], 1):
            dispatcher.utter_message(
                text=f"{i}. {course['nome']} ({course['nivel']})",
                buttons=[
                    {
                        "title": "🔍 Ver Detalhes",
                        "payload": f'/get_course_details{{"course_name":"{course["nome"]}"}}'
                    },
                    {
                        "title": "🌐 Abrir Página",
                        "url": course['url'],
                        "type": "web_url"
                    }
                ]
            )
        
        if len(courses) > 5:
            dispatcher.utter_message(
                text=f"Mostrando 5 de {len(courses)} cursos. Pesquise pelo nome para mais."
            )
        
        return [SlotSet("course_type", course_type)]

    def _detect_course_type(self, message: str) -> Optional[str]:
        """Detecta o tipo de curso com base na mensagem"""
        if any(w in message for w in ['licenciatura', 'licenciaturas', 'lic']):
            return 'licenciatura'
        elif any(w in message for w in ['mestrado', 'mestrados', 'mest']):
            return 'mestrado'
        elif any(w in message for w in ['doutoramento', 'doutoramentos', 'dout']):
            return 'doutoramento'
        return None

class ActionGetCourseDetails(Action):
    def name(self) -> Text:
        return "action_get_course_details"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        db = DatabaseManager()
        course_name = tracker.get_slot("course_name") or \
                     next(tracker.get_latest_entity_values("course_name"), None)

        if not course_name:
            dispatcher.utter_message(text="Qual curso deseja ver os detalhes?")
            return []

        course = db.get_curso_details(course_name)
        db.close()

        if not course:
            dispatcher.utter_message(text=f"Curso '{course_name}' não encontrado.")
            return []
        
        descricao = course.get('descricao', 'Descrição não disponível.')
        message = (
            f"📘 <b>{course['nome']}</b> ({course['nivel']})\n\n"
            f"🏛️ <i>{course.get('instituicao', 'UAb')}</i>\n\n"
            f"📝 {descricao}\n\n"
            f"🌐 Mais informações: {course['url']}"
        )

        dispatcher.utter_message(
            text=message,
            buttons=[
                {
                    "title": "Abrir Página do Curso",
                    "url": course['url'],
                    "type": "web_url"
                },
                {
                    "title": "Voltar para lista de cursos",
                    "payload": "/search_courses"
                }
            ]
        )
        
        return [SlotSet("course_name", course['nome']), SlotSet("course_url", course['url'])]
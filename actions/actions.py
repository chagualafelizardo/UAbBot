import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from typing import Dict, Text, Any, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet
from difflib import SequenceMatcher  # Adicione no topo do arquivo

class UAbCourseScraper:
    def __init__(self):
        self.base_url = "https://guiadoscursos.uab.pt/"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.course_types = {
            'licenciatura': ('', 'Licenciatura'),
            'mestrado': ('mestrados', 'Mestrado'),
            'doutoramento': ('doutoramentos', 'Doutoramento')
        }

    def get_courses(self, course_type=None):
        try:
            if course_type and course_type.lower() in self.course_types:
                path, level = self.course_types[course_type.lower()]
                url = urljoin(self.base_url, path)
                return self._scrape_courses(url, level)
            else:
                return self._scrape_all_courses()
        except Exception as e:
            print(f"Error scraping courses: {e}")
            return []

    def _scrape_courses(self, url, level):
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            courses = []
            for article in soup.find_all('article', class_=lambda x: x and 'col-sm-6' in x):
                link = article.find('a')
                if not link:
                    continue
                
                title = link.get('title', '').strip() or link.text.strip()
                if title:
                    courses.append({
                        'nome': title,
                        'nivel': level,
                        'url': urljoin(self.base_url, link['href'])
                    })
            return courses
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return []

    def _scrape_all_courses(self):
        all_courses = []
        for course_type, (path, level) in self.course_types.items():
            url = urljoin(self.base_url, path)
            courses = self._scrape_courses(url, level)
            all_courses.extend(courses)
        return all_courses

#----------------------------------------------------------------------------------------
# Accao para listar os cursos disponiveis
class ActionSearchUAbCourses(Action):
    def name(self) -> Text:
        return "action_search_uab_courses"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        # Extrai entidades
        course_type = next(tracker.get_latest_entity_values("course_type"), None)
        course_name = next(tracker.get_latest_entity_values("course_name"), None)
        
        scraper = UAbCourseScraper()
        
        if course_name:
            # Busca específica por nome do curso
            all_courses = scraper.get_courses()
            matched_courses = [c for c in all_courses if course_name.lower() in c['nome'].lower()]
            
            if matched_courses:
                course = matched_courses[0]
                dispatcher.utter_message(
                    response="utter_specific_course_info",
                    course_name=course['nome'],
                    course_level=course['nivel'],
                    course_url=course['url']
                )
            else:
                dispatcher.utter_message(response="utter_course_not_found")
                courses_to_show = all_courses[:10]  # Mostra os 10 primeiros
                self._send_courses_list(dispatcher, "todos", courses_to_show)
                
            return [SlotSet("course_name", course_name if matched_courses else None),
                    SlotSet("course_type", matched_courses[0]['nivel'].lower() if matched_courses else None)]
        
        elif course_type:
            # Busca por tipo de curso
            courses = scraper.get_courses(course_type)
            if courses:
                self._send_courses_list(dispatcher, course_type, courses)
            else:
                dispatcher.utter_message(text=f"Não encontrei cursos do tipo {course_type}.")
            return [SlotSet("course_type", course_type)]
        
        else:
            # Sem entidades específicas - pede mais informações
            dispatcher.utter_message(response="utter_ask_course_type")
            return []

    def _send_courses_list(self, dispatcher, course_type, courses):
        formatted = "\n".join(
            f"- {i+1}. {c['nome']} ({c['nivel']})" 
            for i, c in enumerate(courses[:10])  # Limita a 10 cursos
        )
        
        if len(courses) > 10:
            formatted += f"\n\nMostrando 10 de {len(courses)} cursos encontrados."
        
        dispatcher.utter_message(
            response="utter_courses_list",
            course_type=course_type,
            formatted_courses=formatted
        )

#----------------------------------------------------------------------------------------
# Nava accao para lidar com os detalhes dos cursos

class ActionGetCourseDetails(Action):
    def name(self) -> Text:
        return "action_get_course_details"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        # Extrai nome do curso do comando especial
        course_name = tracker.latest_message.get("text", "").split('course_name":"')[-1].rstrip('"}')
        
        if not course_name:
            dispatcher.utter_message(text="Não consegui identificar qual curso você quer ver os detalhes.")
            return []

        scraper = UAbCourseScraper()
        all_courses = scraper.get_courses()
        
        # Busca exata (case insensitive)
        matched_courses = [c for c in all_courses if course_name.lower() in c['nome'].lower()]
        
        if not matched_courses:
            # Tenta encontrar cursos similares
            similar_courses = [c for c in all_courses if fuzz.partial_ratio(course_name.lower(), c['nome'].lower()) > 80]
            if similar_courses:
                matched_courses = similar_courses[:1]  # Pega o mais similar
            else:
                dispatcher.utter_message(text=f"Não encontrei o curso '{course_name}'. Aqui está nossa lista de cursos:")
                self._send_courses_list(dispatcher, "todos", all_courses[:5])  # Mostra os 5 primeiros
                return []

        course = matched_courses[0]
        
        try:
            # Tenta obter detalhes do curso
            details = self._get_course_details(course['url'])
            
            # Formata a mensagem com os detalhes
            message = (
                f"📚 <b>{course['nome']}</b> ({course['nivel']})\n\n"
                f"🔗 <a href='{course['url']}' target='_blank'>Página oficial</a>\n\n"
                f"📝 <b>Descrição:</b> {details.get('description', 'Informação não disponível')}\n\n"
                f"⏳ <b>Duração:</b> {details.get('duration', 'Informação não disponível')}\n"
                f"🎓 <b>Coordenação:</b> {details.get('coordinator', 'Informação não disponível')}\n"
                f"📋 <b>Requisitos:</b> {details.get('requirements', 'Informação não disponível')}\n"
            )
            
            dispatcher.utter_message(text=message)
            
        except Exception as e:
            print(f"Error getting course details: {e}")
            dispatcher.utter_message(
                text=f"Aqui estão as informações básicas sobre {course['nome']}:\n\n"
                     f"Tipo: {course['nivel']}\n"
                     f"Mais informações: {course['url']}"
            )
        
        return [SlotSet("course_name", course['nome'])]
        
#----------------------------------------------------------------------------------------
# 
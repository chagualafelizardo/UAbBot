import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from typing import Dict, Text, Any, List, Optional
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet

class UAbCourseScraper:
    def __init__(self):
        self.base_url = "https://guiadoscursos.uab.pt/"
        self.session = requests.Session()
        self.session.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

    def get_courses(self, course_type: Optional[str] = None) -> List[Dict[str, str]]:
        """Obtém cursos com base no tipo especificado"""
        if not course_type:
            return self._get_all_courses()
            
        if course_type == 'licenciatura':
            return self._get_licenciaturas()
        elif course_type == 'mestrado':
            return self._get_mestrados()
        elif course_type == 'doutoramento':
            return self._get_doutoramentos()
        else:
            return []

    def _get_all_courses(self) -> List[Dict[str, str]]:
        """Obtém todos os cursos de todos os tipos"""
        all_courses = []
        all_courses.extend(self._get_licenciaturas())
        all_courses.extend(self._get_mestrados())
        all_courses.extend(self._get_doutoramentos())
        return all_courses

    def _get_licenciaturas(self) -> List[Dict[str, str]]:
        """Obtém cursos de licenciatura"""
        url = self.base_url
        print(f"DEBUG: Acessando licenciaturas em: {url}")
        
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return self._parse_licenciaturas(response.text)
        except Exception as e:
            print(f"ERRO ao acessar licenciaturas: {str(e)}")
            return []

    def _get_mestrados(self) -> List[Dict[str, str]]:
        """Obtém cursos de mestrado"""
        url = urljoin(self.base_url, "mestrados/")
        print(f"DEBUG: Acessando mestrados em: {url}")
        
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return self._parse_mestrados(response.text)
        except Exception as e:
            print(f"ERRO ao acessar mestrados: {str(e)}")
            return []

    def _get_doutoramentos(self) -> List[Dict[str, str]]:
        """Obtém cursos de doutoramento"""
        url = urljoin(self.base_url, "doutoramentos/")
        print(f"DEBUG: Acessando doutoramentos em: {url}")
        
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return self._parse_doutoramentos(response.text)
        except Exception as e:
            print(f"ERRO ao acessar doutoramentos: {str(e)}")
            return []

    def _parse_licenciaturas(self, html: str) -> List[Dict[str, str]]:
        """Analisa o HTML das licenciaturas"""
        soup = BeautifulSoup(html, 'html.parser')
        courses = []
        
        # Para licenciaturas
        cards = soup.find_all('article', class_=lambda x: x and 'col-sm-6' in x)
        
        for card in cards:
            link = card.find('a')
            if not link:
                continue
                
            title = link.get('title', '').strip() or link.text.strip()
            if not title:
                continue
                
            courses.append({
                'nome': title,
                'nivel': 'Licenciatura',
                'url': urljoin(self.base_url, link['href'])
            })
            
        print(f"DEBUG: Encontradas {len(courses)} licenciaturas")
        return courses

    def _parse_mestrados(self, html: str) -> List[Dict[str, str]]:
        """Analisa o HTML dos mestrados"""
        soup = BeautifulSoup(html, 'html.parser')
        courses = []
        
        # Para mestrados
        cards = soup.find_all('article', class_='col-sm-6 col-md-4 col-lg-3')
        
        for card in cards:
            link = card.find('a')
            if not link:
                continue
                
            title_element = card.find('h4', class_='courses-text')
            if not title_element:
                continue
                
            title = title_element.get_text(strip=True)
            if not title:
                continue
                
            courses.append({
                'nome': title,
                'nivel': 'Mestrado',
                'url': urljoin(self.base_url, link['href'])
            })
            
        print(f"DEBUG: Encontrados {len(courses)} mestrados")
        return courses

    def _parse_doutoramentos(self, html: str) -> List[Dict[str, str]]:
        """Analisa o HTML dos doutoramentos"""
        soup = BeautifulSoup(html, 'html.parser')
        courses = []
        
        # Para doutoramentos
        cards = soup.find_all('article', class_='col-sm-6 col-md-4 col-lg-3')
        
        for card in cards:
            link = card.find('a')
            if not link:
                continue
                
            title_element = card.find('h4', class_='courses-text')
            if not title_element:
                continue
                
            title = title_element.get_text(strip=True)
            if not title:
                continue
                
            courses.append({
                'nome': title,
                'nivel': 'Doutoramento',
                'url': urljoin(self.base_url, link['href'])
            })
            
        print(f"DEBUG: Encontrados {len(courses)} doutoramentos")
        return courses
    
# -------------------------------------------------------------------
class ActionSearchUAbCourses(Action):
    def name(self) -> Text:
        return "action_search_uab_courses"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        message = tracker.latest_message.get('text', '').lower()
        course_type = self._detect_course_type(message)
        
        print(f"\n[DEBUG] Mensagem: {message}")
        print(f"[DEBUG] Tipo detectado: {course_type}\n")

        if not course_type:
            dispatcher.utter_message(response="utter_ask_course_type")
            return []

        scraper = UAbCourseScraper()
        courses = scraper.get_courses(course_type)

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
                        "payload": f'/get_course_details{{"course_name":"{course["nome"]}","course_url":"{course["url"]}"}}'
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
    
# -------------------------------------------------------------------
class ActionGetCourseDetails(Action):
    def name(self) -> Text:
        return "action_get_course_details"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        # Obtém o nome e URL do curso dos slots ou da mensagem
        course_name = tracker.get_slot("course_name") or \
                     next(tracker.get_latest_entity_values("course_name"), None)
        
        course_url = tracker.get_slot("course_url") or \
                     next(tracker.get_latest_entity_values("course_url"), None)

        if not course_name and not course_url:
            dispatcher.utter_message(text="Qual curso deseja ver os detalhes?")
            return []

        # Se não temos URL, tentamos encontrar pelo nome
        if not course_url:
            scraper = UAbCourseScraper()
            all_courses = scraper.get_courses()
            matched = [c for c in all_courses if course_name and course_name.lower() in c['nome'].lower()]
            
            if not matched:
                dispatcher.utter_message(text=f"Curso '{course_name}' não encontrado.")
                return []
            
            course_url = matched[0]['url']
            course_name = matched[0]['nome']

        # Monta a mensagem com o link direto
        message = (
            f"📘 <b>{course_name}</b>\n\n"
            f"🌐 Acesse a página completa do curso para todas as informações:\n"
            f"{course_url}"
        )

        dispatcher.utter_message(
            text=message,
            buttons=[
                {
                    "title": "Abrir Página do Curso",
                    "url": course_url,
                    "type": "web_url"
                },
                {
                    "title": "Voltar para lista de cursos",
                    "payload": "/search_courses"
                }
            ]
        )
        
        return [SlotSet("course_name", course_name), SlotSet("course_url", course_url)]
    
# -------------------------------------------------------------------
# Informacao sobre a uab ...
class ActionBuscarInfoUAb(Action):
    def name(self) -> str:
        return "action_buscar_info_uab"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[str, Any]) -> List[Dict[str, Any]]:

        # Tenta obter o tópico de diferentes formas
        topico = tracker.get_slot("topico")
        
        # Se não estiver no slot, tenta pegar da última mensagem do usuário
        if not topico:
            last_user_message = next(
                (event.get('text') for event in reversed(tracker.events) 
                 if event.get('event') == 'user'),
                None
            )
            if last_user_message:
                topico = last_user_message.lower().replace("fala-me sobre", "").strip()
        
        # Se ainda não encontrou, pede para especificar
        if not topico:
            dispatcher.utter_message(text="Por favor, diga qual tópico deseja saber sobre a UAb.")
            return []

        # Mapeamento de tópicos para URLs ou seções específicas
        topicos_uab = {
            "história": "A UAb",
            "fundação": "A UAb",
            "reitoria": "Reitoria",
            "organização": "Organização",
            "honoris causa": "Doutorados Honoris Causa",
            # Adicione outros mapeamentos conforme necessário
        }

        # Normaliza o tópico
        topico_normalizado = topicos_uab.get(topico.lower(), topico)

        scraper = UAbInfoScraper()
        info = scraper.get_uab_info(topico_normalizado)

        if info and info.get("content"):
            resposta = (
                f"📘 **{info['title']}**\n\n"
                f"{info['content']}\n\n"
                f"🔗 Mais detalhes: {info['url']}"
            )
            dispatcher.utter_message(text=resposta)
        else:
            dispatcher.utter_message(text=f"Desculpe, não encontrei informações sobre '{topico}'. Posso ajudar com informações sobre: história, reitoria, organização, ou outros tópicos da UAb.")

        return []
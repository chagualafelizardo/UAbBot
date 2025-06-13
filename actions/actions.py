from typing import Dict, Text, Any, List, Optional
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet
from pymongo import MongoClient
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class MongoDBManager:
    def __init__(self):
        self.client = MongoClient('mongodb://root:root@uabbot-mongodb-1:27017/')
        self.db = self.client['uab']
        self.cursos_collection = self.db['cursos']

    def get_cursos(self, nivel: str = None) -> List[Dict]:
        """Busca cursos por nível (licenciatura, mestrado, doutoramento)"""
        query = {}
        if nivel:
            # Busca tanto no campo 'tipo' quanto no 'nome' do curso
            query['$or'] = [
                {'tipo': nivel},
                {'nome': {'$regex': nivel, '$options': 'i'}}
            ]
        
        projection = {
            'nome': 1,
            'departamento': 1,
            'regulamentacao.regime': 1,
            'regulamentacao.lingua': 1,
            'estrutura_curricular.maior.ects': 1,
            'apresentacao.video.url': 1,
            '_id': 0
        }
        
        try:
            cursos = list(self.cursos_collection.find(query, projection).limit(10))
            
            # Se não encontrar, tenta uma busca mais ampla
            if not cursos and nivel:
                cursos = list(self.cursos_collection.find(
                    {'nome': {'$regex': '|'.join(self._get_keywords_for_level(nivel)), '$options': 'i'}},
                    projection
                ).limit(10))
                
            return cursos
        except Exception as e:
            logger.error(f"Erro ao buscar cursos: {str(e)}")
            return []

    def _get_keywords_for_level(self, nivel: str) -> List[str]:
        """Retorna palavras-chave para cada nível de curso"""
        mapping = {
            'licenciatura': ['licenciatura', 'graduação', '1° ciclo'],
            'mestrado': ['mestrado', 'pós-graduação', '2° ciclo'],
            'doutoramento': ['doutoramento', 'phd', '3° ciclo']
        }
        return mapping.get(nivel.lower(), [nivel])

    def get_curso_details(self, course_name: str) -> Optional[Dict]:
        """Busca detalhes completos de um curso específico"""
        try:
            curso = self.cursos_collection.find_one(
                {'nome': {'$regex': course_name, '$options': 'i'}},
                {'_id': 0}
            )
            
            if not curso:
                return None
                
            # Formata os dados para resposta
            formatted_curso = {
                'nome': curso.get('nome', ''),
                'nivel': curso.get('tipo', ''),
                'descricao': curso.get('descricao', 'Descrição não disponível'),
                'url': curso.get('apresentacao', {}).get('video', {}).get('url', '#'),
                'departamento': curso.get('departamento', 'Departamento não especificado'),
                'regime': curso.get('regulamentacao', {}).get('regime', ''),
                'publico_alvo': "\n".join(f"• {item}" for item in curso.get('publico_alvo', [])),
                'coordenacao': f"{curso.get('coordenacao', {}).get('coordenador', '')} (Coordenador)\n"
                               f"{curso.get('coordenacao', {}).get('vice_coordenador', '')} (Vice-Coordenador)",
                'ects_total': curso.get('estrutura_curricular', {}).get('maior', {}).get('ects', 0),
                'minors': "\n".join(
                    f"• {minor['nome']} ({minor['ects']} ECTS)" 
                    for minor in curso.get('estrutura_curricular', {}).get('minors', [])
                )
            }
            
            return formatted_curso
        except Exception as e:
            logger.error(f"Erro ao buscar detalhes do curso: {str(e)}")
            return None

    def close(self):
        self.client.close()

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
            db = MongoDBManager()
            message = tracker.latest_message.get('text', '').lower()
            logger.info(f"Mensagem do usuário: {message}")

            # Detecção do tipo de curso
            course_type = self._detect_course_type(message)
            if not course_type:
                dispatcher.utter_message(response="utter_ask_course_type")
                return []

            logger.info(f"Buscando cursos do tipo: {course_type}")
            
            # Consulta ao MongoDB
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
                f"• {course['nome']} ({course['departamento']}) - {course['regulamentacao']['regime']}"
                for course in courses[:5]
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
        message = message.lower()
        mapping = {
            'licenciatura': ['licenciatura', 'licenciaturas', 'graduação', '1º ciclo', 'primeiro ciclo'],
            'mestrado': ['mestrado', 'mestrados', 'pós-graduação', '2º ciclo', 'segundo ciclo'],
            'doutoramento': ['doutoramento', 'doutoramentos', '3º ciclo', 'terceiro ciclo', 'phd']
        }
        for course_type, keywords in mapping.items():
            for keyword in keywords:
                if keyword in message:
                    return course_type
        return None

class ActionGetCourseDetails(Action):
    def name(self) -> Text:
        return "action_get_course_details"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        db = None
        try:
            db = MongoDBManager()
            course_name = tracker.get_slot("course_name")

            if not course_name:
                dispatcher.utter_message(text="Por favor, diga o nome do curso que deseja ver.")
                return []

            logger.info(f"Buscando detalhes do curso: {course_name}")

            course = db.get_curso_details(course_name)

            if not course:
                dispatcher.utter_message(
                    text=f"Não encontrei detalhes para o curso '{course_name}'.",
                    buttons=[{
                        "title": "Ver todos os cursos",
                        "payload": "/ask_uab_courses"
                    }]
                )
                return []

            # Formatação rica da resposta
            response = f"""
    📚 *{course['nome']}* ({course['nivel'].capitalize()})

    🏛️ *Departamento:* {course['departamento']}
    📝 *ECTS Totais:* {course['ects_total']}
    🌐 *Regime:* {course['regime']}

    📖 *Descrição:*
    {course['descricao']}

    👥 *Público-Alvo:*
    {course['publico_alvo']}

    👨‍🏫 *Coordenação:*
    {course['coordenacao']}

    🎓 *Minors Disponíveis:*
    {course['minors']}

    🔗 [Acessar página oficial]({course['url']})
    """

            dispatcher.utter_message(
                text=response,
                buttons=[
                    {
                        "title": "Ver disciplinas",
                        "payload": f"/ask_course_subjects{{\"course_name\":\"{course['nome']}\"}}"
                    },
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

class ActionGetCourseSubjects(Action):
    def name(self) -> Text:
        return "action_get_course_subjects"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        db = None
        try:
            db = MongoDBManager()
            course_name = tracker.get_slot("course_name")

            if not course_name:
                dispatcher.utter_message(text="De qual curso você quer ver as disciplinas?")
                return []

            # Busca apenas as disciplinas do curso
            curso = db.cursos_collection.find_one(
                {'nome': {'$regex': course_name, '$options': 'i'}},
                {'estrutura_curricular.disciplinas': 1, 'nome': 1, '_id': 0}
            )

            if not curso or 'estrutura_curricular' not in curso or 'disciplinas' not in curso['estrutura_curricular']:
                dispatcher.utter_message(
                    text=f"Não encontrei disciplinas para o curso '{course_name}'.",
                    buttons=[{
                        "title": "Ver detalhes do curso",
                        "payload": f'/get_course_details{{"course_name":"{course_name}"}}'
                    }]
                )
                return []

            disciplinas = curso['estrutura_curricular']['disciplinas']
            
            # Agrupa por ano e semestre
            anos = {}
            for disciplina in disciplinas:
                ano_key = f"{disciplina['ano']}º Ano - {disciplina['semestre']}"
                if ano_key not in anos:
                    anos[ano_key] = []
                anos[ano_key].append(disciplina)

            # Formata a resposta
            response = f"📚 *Disciplinas de {curso['nome']}*\n\n"
            for ano, disciplinas_ano in anos.items():
                response += f"*{ano}:*\n"
                for disciplina in disciplinas_ano[:5]:  # Limita a 5 disciplinas por semestre
                    response += f"• {disciplina['codigo']} - {disciplina['nome']} ({disciplina['ects']} ECTS)\n"
                response += "\n"

            if len(disciplinas) > 10:
                response += f"ℹ️ Mostrando 10 de {len(disciplinas)} disciplinas. Consulte o plano de estudos completo no site."

            dispatcher.utter_message(
                text=response,
                buttons=[
                    {
                        "title": "Ver detalhes do curso",
                        "payload": f'/get_course_details{{"course_name":"{course_name}"}}'
                    },
                    {
                        "title": "Outros cursos",
                        "payload": "/ask_uab_courses"
                    }
                ]
            )

            return []
        
        except Exception as e:
            logger.error(f"Erro ao buscar disciplinas: {str(e)}", exc_info=True)
            dispatcher.utter_message(
                text="Não consegui acessar as disciplinas deste curso no momento."
            )
            return []
        finally:
            if db:
                db.close()
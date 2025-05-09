from typing import Dict, Text, Any, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

class ActionSearchProgrammingLanguage(Action):
    def name(self) -> Text:
        return "action_search_programming_language"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        language = tracker.get_slot("language")
        if not language:
            dispatcher.utter_message(text="Por favor, especifique uma linguagem de programação.")
            return []
        
        searcher = ProgrammingLanguageSearcher()
        definition, source = searcher.search_definition(language)
        
        if definition:
            message = f"📚 Definição de {language}:\n{definition}\n\n🔍 Fonte: {source}"
        else:
            message = f"Não encontrei uma definição para {language}. Tente verificar a ortografia ou usar o nome em inglês."
        
        dispatcher.utter_message(text=message)
        return []
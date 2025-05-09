# No seu chatbot Rasa, adicione esta ação:
from typing import Dict, Text, Any
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import requests

class ActionSearchCapital(Action):
    def name(self) -> Text:
        return "action_search_capital"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        country = tracker.get_slot("country")
        if not country:
            dispatcher.utter_message(text="Por favor, especifique um país.")
            return []
        
        searcher = OnlineCapitalSearcher()
        capital = searcher.search_capital(country)
        
        if capital:
            dispatcher.utter_message(text=f"A capital de {country} é {capital}")
        else:
            dispatcher.utter_message(text=f"Não encontrei a capital para {country}")
        
        return []
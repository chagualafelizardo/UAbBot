import requests
from typing import Optional

class OnlineCapitalSearcher:
    def __init__(self):
        self.base_url = "https://restcountries.com/v3.1"
    
    def search_capital(self, country_name: str) -> Optional[str]:
        """Procura a capital de um país usando API web"""
        try:
            # Primeiro tentamos encontrar por nome completo
            response = requests.get(f"{self.base_url}/name/{country_name}?fullText=true")
            
            # Se não encontrar, tentamos pesquisa parcial
            if response.status_code == 404:
                response = requests.get(f"{self.base_url}/name/{country_name}")
            
            if response.status_code == 200:
                data = response.json()
                return self.extract_capital(data)
            
            return None
            
        except Exception as e:
            print(f"Erro na pesquisa: {str(e)}")
            return None
    
    def extract_capital(self, api_data: list) -> Optional[str]:
        """Extrai a capital dos dados da API"""
        try:
            if not api_data:
                return None
                
            country_data = api_data[0]
            
            # Alguns países têm múltiplas capitais (ex: África do Sul)
            capitals = country_data.get('capital', ['N/A'])
            return ', '.join(capitals) if isinstance(capitals, list) else capitals
            
        except Exception as e:
            print(f"Erro ao processar dados da API: {str(e)}")
            return None

def main():
    print("=== Pesquisador Online de Capitais ===")
    print("Digite o nome de um país para descobrir sua capital")
    print("Digite 'sair' para encerrar\n")
    
    searcher = OnlineCapitalSearcher()
    
    while True:
        user_input = input("País: ").strip()
        
        if user_input.lower() in ('sair', 'exit', 'quit'):
            break
            
        if not user_input:
            print("Por favor, digite um nome de país.")
            continue
            
        capital = searcher.search_capital(user_input)
        
        if capital:
            print(f"A capital de {user_input.title()} é {capital}\n")
        else:
            print(f"País não encontrado: {user_input}")
            print("Certifique-se de digitar o nome em português ou inglês\n")

if __name__ == "__main__":
    main()
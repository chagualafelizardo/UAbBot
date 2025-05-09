import wikipedia
import requests
from bs4 import BeautifulSoup
from typing import Dict, Optional, Tuple
import urllib.parse

class ProgrammingLanguageSearcher:
    def __init__(self):
        wikipedia.set_lang("pt")  # Configura para português
        
    def search_definition(self, language_name: str) -> Tuple[Optional[str], Optional[str]]:
        """Busca definição da linguagem de programação
        Retorna: (definição_resumida, fonte_completa)
        """
        try:
            # Tenta primeiro com Wikipedia
            wiki_summary = wikipedia.summary(f"{language_name} (linguagem de programação)", sentences=3)
            wiki_url = wikipedia.page(f"{language_name} (linguagem de programação)").url
            return (wiki_summary, wiki_url)
            
        except wikipedia.exceptions.DisambiguationError as e:
            # Caso haja ambiguidade no termo
            try:
                option = e.options[0]
                wiki_summary = wikipedia.summary(option, sentences=3)
                wiki_url = wikipedia.page(option).url
                return (wiki_summary, wiki_url)
            except:
                pass  # Vamos tentar a busca web
                
        except wikipedia.exceptions.PageError:
            pass  # Vamos tentar a busca web
            
        # Se Wikipedia falhar, tenta busca web genérica
        return self._web_search(language_name)
    
    def _web_search(self, language_name: str) -> Tuple[Optional[str], Optional[str]]:
        """Busca alternativa em sites de tecnologia"""
        try:
            query = urllib.parse.quote_plus(f"{language_name} linguagem de programação site:pt.wikipedia.org OR site:developer.mozilla.org OR site:devdocs.io")
            url = f"https://www.google.com/search?q={query}"
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Tenta pegar o primeiro resultado (snippet)
            result = soup.find('div', class_='VwiC3b')
            if result:
                source_link = soup.find('a', href=True)['href']
                return (result.get_text(), source_link)
                
            return (None, None)
            
        except Exception as e:
            print(f"Erro na busca web: {str(e)}")
            return (None, None)

def main():
    print("=== Pesquisador de Linguagens de Programação ===")
    print("Digite o nome de uma linguagem para obter sua definição")
    print("Exemplos: Python, Java, C++, JavaScript")
    print("Digite 'sair' para encerrar\n")
    
    searcher = ProgrammingLanguageSearcher()
    
    while True:
        user_input = input("Linguagem: ").strip()
        
        if user_input.lower() in ('sair', 'exit', 'quit'):
            break
            
        if not user_input:
            print("Por favor, digite o nome de uma linguagem.")
            continue
            
        definition, source = searcher.search_definition(user_input)
        
        if definition:
            print(f"\nDefinição de {user_input}:")
            print(definition)
            print(f"\nFonte completa: {source}\n")
        else:
            print(f"\nNão foi possível encontrar uma definição para {user_input}")
            print("Sugestões:")
            print("- Verifique a ortografia")
            print("- Tente o nome em inglês (ex: 'JavaScript' em vez de 'Javascript')")
            print("- Algumas linguagens menos conhecidas podem não ter resultados\n")

if __name__ == "__main__":
    main()
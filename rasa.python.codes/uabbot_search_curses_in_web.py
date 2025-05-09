import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

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
            'doutoramento': ('doutoramentos', 'Doutoramento'),
            'todos': None  # Caso especial para todos os cursos
        }

    def scrape_courses_from_page(self, url, course_level):
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            course_articles = soup.find_all('article', class_=lambda x: x and 'col-sm-6' in x)
            courses = []
            
            for article in course_articles:
                link_tag = article.find('a')
                if not link_tag:
                    continue
                    
                link = urljoin(self.base_url, link_tag['href'])
                title = link_tag.get('title', '').strip()
                
                img_tag = article.find('img')
                img_src = img_tag['src'] if img_tag else ''
                
                courses.append({
                    'nivel': course_level,
                    'titulo': title,
                    'area': "",
                    'descricao': "",
                    'url': link,
                    'imagem': img_src
                })
            
            return courses

        except Exception as e:
            print(f"Erro ao acessar {url}: {str(e)}")
            return []

    def scrape_courses_by_type(self, course_type):
        course_type = course_type.lower()
        if course_type not in self.course_types:
            print(f"Tipo de curso inválido. Opções válidas: {', '.join(self.course_types.keys())}")
            return []
        
        if course_type == 'todos':
            return self.scrape_all_courses()
        
        path, level = self.course_types[course_type]
        url = urljoin(self.base_url, path) if path else self.base_url
        
        print(f"\nColetando {level}s de {url}...")
        courses = self.scrape_courses_from_page(url, level)
        print(f"Encontrados {len(courses)} {level}s")
        
        return courses

    def scrape_all_courses(self):
        all_courses = []
        
        for course_type, (path, level) in filter(lambda x: x[1] is not None, self.course_types.items()):
            url = urljoin(self.base_url, path) if path else self.base_url
            print(f"\nColetando {level}s de {url}...")
            
            courses = self.scrape_courses_from_page(url, level)
            all_courses.extend(courses)
            
            print(f"Encontrados {len(courses)} {level}s")
        
        return all_courses

def get_user_input():
    print("\nTipos de cursos disponíveis:")
    print("- Licenciatura")
    print("- Mestrado")
    print("- Doutoramento")
    print("- Todos")

    while True:
        choice = input("\nDigite o nome do tipo de curso que deseja listar (ou 'sair' para terminar): ").strip().lower()
        
        if choice == 'sair':
            return None
        elif choice in ['licenciatura', 'mestrado', 'doutoramento', 'todos']:
            return choice
        else:
            print("Opção inválida. Por favor, digite 'Licenciatura', 'Mestrado', 'Doutoramento', 'Todos' ou 'sair'.")


if __name__ == "__main__":
    scraper = UAbCourseScraper()
    
    while True:
        course_type = get_user_input()
        if course_type is None:
            print("Programa encerrado.")
            break
        
        print(f"\nIniciando scraping de cursos do tipo: {course_type}...")
        courses = scraper.scrape_courses_by_type(course_type)
        
        if not courses:
            print("\nNão foi possível encontrar cursos. Verifique se a estrutura do site mudou.")
        else:
            print(f"\nTotal de cursos encontrados: {len(courses)}")
            for i, course in enumerate(courses, 1):
                print(f"\n{i}. {course['nivel']} - {course['titulo']}")
                print(f"URL: {course['url']}")
        
        input("\nPressione Enter para continuar ou Ctrl+C para sair...")
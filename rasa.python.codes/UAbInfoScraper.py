import requests
from bs4 import BeautifulSoup
from typing import Dict, Optional

class UAbInstitutionScraper:
    BASE_URL = "https://portal.uab.pt/conhecer-a-uab/ "
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }

    def __init__(self):
        self.soup = self._get_soup()

    def _get_soup(self) -> Optional[BeautifulSoup]:
        """Obtém e parseia o conteúdo HTML da página"""
        try:
            response = requests.get(self.BASE_URL, headers=self.HEADERS)
            response.raise_for_status()
            return BeautifulSoup(response.text, 'html.parser')
        except requests.RequestException as e:
            print(f"Erro ao carregar a página: {e}")
            return None

    def get_institution_info(self, topic: str) -> Optional[Dict[str, str]]:
        """Retorna informações institucionais com base no tópico solicitado"""
        if not self.soup:
            return None

        topic_methods = {
            'historia': self._scrape_history,
            'missao': self._scrape_mission,
            'contactos': self._scrape_contacts,
            'reitoria': self._scrape_reitoria,
            'organizacao': self._scrape_organization,
            'geral': self._scrape_general_info
        }

        method = topic_methods.get(topic.lower())
        if method:
            return method()
        else:
            print(f"Tópico '{topic}' não suportado.")
            return None

    def _scrape_section_content(self, title_text: str, next_tag='p') -> str:
        """Função auxiliar para extrair conteúdo após um título específico"""
        section = self.soup.find('h2', string=title_text)
        if section and section.find_next(next_tag):
            return section.find_next(next_tag).get_text(strip=True)
        return ""

    def _scrape_general_info(self) -> Dict[str, str]:
        """Extrai informações gerais sobre a UAb"""
        content = self._scrape_section_content("A UAb")
        return {
            'title': 'Sobre a Universidade Aberta',
            'content': content or "Informações gerais não encontradas.",
            'url': self.BASE_URL
        }

    def _scrape_history(self) -> Dict[str, str]:
        """Extrai informações históricas da UAb"""
        content = self._scrape_section_content("História")
        return {
            'title': 'História da UAb',
            'content': content or "História não encontrada.",
            'url': self.BASE_URL + "#historia"
        }

    def _scrape_mission(self) -> Dict[str, str]:
        """Extrai a missão e valores da UAb"""
        content = self._scrape_section_content("Missão e Valores")
        return {
            'title': 'Missão e Valores da UAb',
            'content': content or "Missão e valores não encontrados.",
            'url': self.BASE_URL + "#missao-e-valores"
        }

    def _scrape_reitoria(self) -> Dict[str, str]:
        """Extrai informações sobre a reitoria"""
        reitoria_div = self.soup.find('div', class_='entry-content')
        if reitoria_div:
            reitoria_p = reitoria_div.find('p', string=lambda text: text and 'Reitora' in text)
            content = reitoria_p.get_text(strip=True) if reitoria_p else ""
        else:
            content = ""

        return {
            'title': 'Reitoria da UAb',
            'content': content or "Informações sobre a reitoria não encontradas.",
            'url': self.BASE_URL + "#reitoria"
        }

    def _scrape_organization_link(self) -> Dict[str, str]:
        """Extrai link para a seção de organização"""
        organization_link = self.soup.find('a', string='Organização')
        if organization_link and 'href' in organization_link.attrs:
            full_url = urljoin(self.BASE_URL, organization_link['href'])
            return {
                'title': 'Organização da UAb',
                'content': f"Acesse a página completa em: {full_url}",
                'url': full_url
            }
        return {
            'title': 'Organização da UAb',
            'content': 'Link para organização não encontrado.',
            'url': self.BASE_URL
        }

    def _scrape_contacts(self) -> Dict[str, str]:
        """Extrai informações de contato da UAb"""
        footer = self.soup.find('footer', class_='site-footer')
        if footer:
            contact_info = footer.get_text(strip=True)
            return {
                'title': 'Contactos da UAb',
                'content': contact_info[:500] + '...' if len(contact_info) > 500 else contact_info,
                'url': "https://portal.uab.pt/contactos/ "
            }
        return {
            'title': 'Contactos da UAb',
            'content': 'Não foram encontradas informações de contacto.',
            'url': "https://portal.uab.pt/contactos/ "
        }
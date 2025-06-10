import sys
import os
import time
import logging

# Adiciona caminho relativo para importar actions.db_manager
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from actions.db_manager import DatabaseManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Substitui o scraping por dados locais
def get_mocked_courses():
    return [
        {
            "nome": "Licenciatura em Ciências da Educação",
            "nivel": "Licenciatura",
            "url": "https://www.uab.pt/curso/educacao"
        },
        {
            "nome": "Mestrado em Administração Pública",
            "nivel": "Mestrado",
            "url": "https://www.uab.pt/curso/administracao-publica"
        },
        {
            "nome": "Doutoramento em Educação",
            "nivel": "Doutoramento",
            "url": "https://www.uab.pt/curso/doutoramento-educacao"
        }
    ]

def populate_courses():
    max_retries = 5
    retry_delay = 5

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Tentativa {attempt}/{max_retries} de popular banco de dados")

            db = DatabaseManager()

            logger.info("Limpando tabela cursos...")
            with db.conn.cursor() as cur:
                cur.execute("TRUNCATE TABLE cursos RESTART IDENTITY")
            db.conn.commit()

            logger.info("Obtendo cursos da fonte local...")
            all_courses = get_mocked_courses()
            inserted = 0

            logger.info(f"Encontrados {len(all_courses)} cursos para inserir")

            for course in all_courses:
                try:
                    with db.conn.cursor() as cur:
                        cur.execute("""
                        INSERT INTO cursos (nome, nivel, url)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (nome, nivel) DO NOTHING
                        """, (course['nome'], course['nivel'], course['url']))
                        if cur.rowcount > 0:
                            inserted += 1
                except Exception as e:
                    logger.error(f"Erro ao inserir curso {course['nome']}: {e}")

            db.conn.commit()
            db.close()
            logger.info(f"✅ Inseridos {inserted}/{len(all_courses)} cursos no banco de dados")
            return

        except Exception as e:
            logger.error(f"Tentativa {attempt} falhou: {str(e)}")
            if attempt < max_retries:
                time.sleep(retry_delay)
                continue
            logger.error("❌ Falha crítica ao popular banco de dados")
            raise

if __name__ == "__main__":
    populate_courses()

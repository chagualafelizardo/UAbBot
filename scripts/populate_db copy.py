import sys
import os
import time
import logging
from typing import List, Dict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from actions.db_manager import DatabaseManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_real_courses_from_source() -> List[Dict[str, str]]:
    """Substitua esta função pela sua fonte real de dados (API, CSV, etc.)"""
    return [
        {
            "nome": "Licenciatura em Ciências da Educação",
            "nivel": "Licenciatura",
            "url": "https://www.uab.pt/curso/educacao",
            "descricao": "Curso de formação de educadores"
        },
        {
            "nome": "Mestrado em Administração Pública",
            "nivel": "Mestrado",
            "url": "https://www.uab.pt/curso/administracao-publica",
            "descricao": "Formação avançada em gestão pública"
        }
    ]

def populate_courses():
    max_retries = 5
    retry_delay = 5

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Tentativa {attempt}/{max_retries} de popular banco de dados")

            db = DatabaseManager()

            logger.info("Verificando se a tabela cursos existe...")
            with db.conn.cursor() as cur:
                cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'cursos'
                )
                """)
                if not cur.fetchone()[0]:
                    raise Exception("Tabela cursos não existe")

            logger.info("Obtendo cursos da fonte real...")
            all_courses = get_real_courses_from_source()
            
            if not all_courses:
                logger.warning("Nenhum curso encontrado na fonte de dados")
                return

            logger.info(f"Preparando para inserir {len(all_courses)} cursos")
            
            inserted = 0
            for course in all_courses:
                try:
                    with db.conn.cursor() as cur:
                        cur.execute("""
                        INSERT INTO cursos (nome, nivel, url, descricao)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (nome, nivel) DO UPDATE SET
                            url = EXCLUDED.url,
                            descricao = EXCLUDED.descricao
                        """, (
                            course['nome'],
                            course['nivel'],
                            course['url'],
                            course.get('descricao', '')
                        ))
                        if cur.rowcount > 0:
                            inserted += 1
                            logger.debug(f"Inserido/atualizado: {course['nome']}")
                except Exception as e:
                    logger.error(f"Erro ao inserir curso {course['nome']}: {e}")
                    db.conn.rollback()

            db.conn.commit()
            db.close()
            logger.info(f"✅ Inseridos/atualizados {inserted} cursos no banco de dados")
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
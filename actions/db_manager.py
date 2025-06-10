import os
import time
import logging
import psycopg2
from psycopg2 import sql
from psycopg2.extras import DictCursor
from typing import List, Dict, Any, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, max_retries: int = 3, retry_delay: int = 2):
        self.conn_params = self._get_connection_params()
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.conn = None
        self.connect()

    def _get_connection_params(self) -> Dict[str, Any]:
        """Valida e retorna parâmetros de conexão"""
        params = {
            "host": os.getenv("DB_HOST", "postgres"),
            "port": int(os.getenv("DB_PORT", "5432")),
            "dbname": os.getenv("DB_NAME", "rasa"),
            "user": os.getenv("DB_USER", "rasa"),
            "password": os.getenv("DB_PASSWORD", "rasa"),
            "connect_timeout": 5
        }
        
        logger.info(f"Conectando ao PostgreSQL em {params['host']}:{params['port']}")
        return params

    def connect(self):
        """Estabelece conexão com tratamento de erros robusto"""
        for attempt in range(1, self.max_retries + 1):
            try:
                self.conn = psycopg2.connect(**self.conn_params)
                self.conn.autocommit = False
                logger.info("✅ Conexão com PostgreSQL estabelecida")
                return
            except psycopg2.OperationalError as e:
                logger.warning(f"Tentativa {attempt}/{self.max_retries} falhou: {e}")
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)
                else:
                    raise ConnectionError(f"Falha ao conectar após {self.max_retries} tentativas")

    def get_cursos(self, nivel: Optional[str] = None, search_query: Optional[str] = None) -> List[Dict[str, Any]]:
        """Consulta otimizada com filtros combinados"""
        try:
            logger.info(f"Iniciando consulta - Nivel: {nivel}, Search: {search_query}")
            
            base_query = """
            SELECT 
                id, nome, nivel, url, descricao, instituicao
            FROM cursos
            WHERE 1=1
            """
            
            params = []
            conditions = []
            
            if nivel:
                conditions.append("nivel ILIKE %s")
                params.append(f"%{nivel}%")
                
            if search_query:
                conditions.append("nome ILIKE %s")
                params.append(f"%{search_query}%")
            
            query = base_query
            if conditions:
                query += " AND " + " AND ".join(conditions)
            
            query += " ORDER BY nome LIMIT 20"
            
            logger.info(f"Query final: {query}")
            logger.info(f"Parâmetros: {params}")
            
            with self.conn.cursor(cursor_factory=DictCursor) as cur:
                cur.execute(query, params)
                results = [dict(row) for row in cur.fetchall()]
                logger.info(f"Resultados encontrados: {len(results)}")
                return results
                
        except Exception as e:
            logger.error(f"Erro detalhado na consulta: {str(e)}", exc_info=True)
            raise

    def get_curso_details(self, curso_nome: str) -> Optional[Dict[str, Any]]:
        """Busca detalhes com fuzzy matching"""
        query = """
        SELECT 
            id, nome, nivel, url, descricao, instituicao,
            similarity(nome, %s) AS score
        FROM cursos
        WHERE nome % %s OR similarity(nome, %s) > 0.3
        ORDER BY score DESC
        LIMIT 1
        """
        
        try:
            with self.conn.cursor(cursor_factory=DictCursor) as cur:
                cur.execute(query, (curso_nome, curso_nome, curso_nome))
                result = cur.fetchone()
                return dict(result) if result else None
        except Exception as e:
            logger.error(f"Erro ao buscar detalhes: {str(e)}")
            raise

    def close(self):
        """Fecha conexão de forma segura"""
        if self.conn and not self.conn.closed:
            self.conn.close()
            logger.info("Conexão com PostgreSQL fechada")
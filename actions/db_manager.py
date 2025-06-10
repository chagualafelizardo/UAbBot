import os
import time
import logging
import psycopg2
from psycopg2 import sql
from psycopg2.extras import DictCursor
from typing import List, Dict, Any, Optional


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, max_retries=5, retry_delay=5):
        self.conn_params = self._get_connection_params()
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.conn = None
        self.connect()

    def _get_connection_params(self) -> Dict[str, Any]:
        """Obtém e valida os parâmetros de conexão do .env"""
        params = {
            "host": os.getenv("DB_HOST", "postgres"),  # Usa nome do serviço como padrão
            "port": int(os.getenv("DB_PORT", "5432")),
            "dbname": os.getenv("DB_NAME"),
            "user": os.getenv("DB_USER"),
            "password": os.getenv("DB_PASSWORD"),
            "connect_timeout": 10  # Adiciona timeout explícito
        }
        
        missing = [k for k, v in params.items() if not v and k != "password"]
        if missing:
            raise ValueError(
                f"Variáveis de ambiente faltando para conexão com PostgreSQL: {missing}\n"
                "Defina-as no arquivo .env"
            )
        
        logger.info(f"Conectando ao PostgreSQL em {params['host']}:{params['port']}")
        return params

    def connect(self):
        """Estabelece conexão com retry automático"""
        for attempt in range(1, self.max_retries + 1):
            try:
                self.conn = psycopg2.connect(**self.conn_params)
                self.conn.autocommit = False
                self._initialize_db()
                logger.info("✅ Conexão com PostgreSQL estabelecida com sucesso")
                return
            except psycopg2.OperationalError as e:
                logger.warning(f"Tentativa {attempt}/{self.max_retries} falhou: {e}")
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)
                else:
                    logger.error("❌ Falha ao conectar ao PostgreSQL após várias tentativas")
                    raise
            except Exception as e:
                logger.error(f"Erro inesperado ao conectar ao PostgreSQL: {e}")
                raise

    def _initialize_db(self):
        """Cria as tabelas necessárias se não existirem"""
        commands = [
            """
            CREATE TABLE IF NOT EXISTS cursos (
                id SERIAL PRIMARY KEY,
                nome VARCHAR(255) NOT NULL,
                nivel VARCHAR(50) NOT NULL,
                url VARCHAR(255) NOT NULL,
                descricao TEXT,
                instituicao VARCHAR(100) DEFAULT 'UAb',
                UNIQUE(nome, nivel)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS conversas (
                id SERIAL PRIMARY KEY,
                sender_id VARCHAR(255) NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                mensagem TEXT NOT NULL,
                intencao VARCHAR(100),
                resposta TEXT,
                contexto JSONB
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_conversas_sender ON conversas(sender_id)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_conversas_timestamp ON conversas(timestamp)
            """
        ]
        
        try:
            with self.conn.cursor() as cur:
                for command in commands:
                    cur.execute(command)
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            print(f"Erro ao inicializar banco de dados: {e}")
            raise

    def get_cursos(self, nivel: Optional[str] = None, search_query: Optional[str] = None) -> List[Dict[str, Any]]:
        """Busca cursos no banco de dados"""
        query = sql.SQL("SELECT nome, nivel, url, descricao FROM cursos WHERE 1=1")
        params = []
        
        if nivel:
            query = sql.SQL("{} AND nivel = %s").format(query)
            params.append(nivel)
            
        if search_query:
            query = sql.SQL("{} AND nome ILIKE %s").format(query)
            params.append(f"%{search_query}%")
            
        query = sql.SQL("{} ORDER BY nome LIMIT 50").format(query)
        
        try:
            with self.conn.cursor(cursor_factory=DictCursor) as cur:
                cur.execute(query, params)
                return [dict(row) for row in cur.fetchall()]
        except Exception as e:
            print(f"Erro ao buscar cursos: {e}")
            return []

    def get_curso_details(self, curso_nome: str) -> Optional[Dict[str, Any]]:
        """Busca detalhes de um curso específico"""
        query = """
        SELECT nome, nivel, url, descricao, instituicao 
        FROM cursos 
        WHERE nome ILIKE %s 
        LIMIT 1
        """
        
        try:
            with self.conn.cursor(cursor_factory=DictCursor) as cur:
                cur.execute(query, (f"%{curso_nome}%",))
                result = cur.fetchone()
                return dict(result) if result else None
        except Exception as e:
            print(f"Erro ao buscar detalhes do curso: {e}")
            return None

    def log_conversa(self, sender_id: str, mensagem: str, intencao: str, resposta: str, contexto: Dict[str, Any]):
        """Registra interações do usuário para análise"""
        query = """
        INSERT INTO conversas (sender_id, mensagem, intencao, resposta, contexto)
        VALUES (%s, %s, %s, %s, %s)
        """
        
        try:
            with self.conn.cursor() as cur:
                cur.execute(query, (sender_id, mensagem, intencao, resposta, contexto))
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            print(f"Erro ao registrar conversa: {e}")

    def close(self):
        if self.conn:
            self.conn.close()
-- Garante privilégios para o usuário padrão postgres
GRANT ALL PRIVILEGES ON DATABASE rasa TO postgres;
ALTER DATABASE rasa OWNER TO postgres;

-- Cria usuário específico da aplicação (opcional)
DO $$
BEGIN
  IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'postgres') THEN
    CREATE USER postgres WITH PASSWORD 'postgres';
    GRANT ALL PRIVILEGES ON DATABASE rasa TO postgres;
  END IF;
END
$$;
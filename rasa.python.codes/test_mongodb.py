from pymongo import MongoClient
import logging
from pprint import pprint

# Configuração básica de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MongoDBTester:
    def __init__(self):
        self.client = None
        self.db = None
        self.collection = None
        
    def connect(self):
        """Tenta conectar ao MongoDB com fallback para credenciais padrão"""
        try:
            # Tentativa sem autenticação
            self.client = MongoClient(
                'mongodb://localhost:27017/',
                serverSelectionTimeoutMS=3000
            )
            self.client.admin.command('ping')
            self.db = self.client['uab']
            logger.info("✅ Conexão bem-sucedida sem autenticação")
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ Falha sem autenticação: {str(e)}")
            try:
                # Tentativa com credenciais padrão
                self.client = MongoClient(
                    'mongodb://root:root@localhost:27017/',
                    authSource='admin',
                    serverSelectionTimeoutMS=3000
                )
                self.client.admin.command('ping')
                self.db = self.client['uab']
                logger.info("✅ Conexão bem-sucedida com autenticação padrão")
                return True
                
            except Exception as e:
                logger.error(f"❌ Falha definitiva na conexão: {str(e)}")
                return False
    
    def test_connection(self):
        """Testa a conexão e lista bancos de dados disponíveis"""
        if not self.connect():
            return False
        
        try:
            # Lista todos os bancos de dados
            db_list = self.client.list_database_names()
            logger.info("\n📦 Bancos de dados disponíveis:")
            for db in db_list:
                print(f"- {db}")
            
            # Verifica se a coleção 'cursos' existe
            if 'uab' in db_list:
                collection_list = self.client['uab'].list_collection_names()
                logger.info("\n🗂 Coleções na base 'uab':")
                for col in collection_list:
                    print(f"- {col}")
                
                if 'cursos' in collection_list:
                    return True
                else:
                    logger.error("Coleção 'cursos' não encontrada na base 'uab'")
                    return False
            else:
                logger.error("Base de dados 'uab' não encontrada")
                return False
                
        except Exception as e:
            logger.error(f"Erro ao listar bancos: {str(e)}")
            return False
    
    def show_sample_data(self):
        """Mostra alguns documentos da coleção cursos"""
        if not self.test_connection():
            return
        
        try:
            self.collection = self.db['cursos']
            
            # Conta documentos
            count = self.collection.count_documents({})
            logger.info(f"\n📊 Total de cursos registrados: {count}")
            
            # Mostra 3 documentos de exemplo
            logger.info("\n🔍 Exemplo de documentos (3 primeiros):")
            for doc in self.collection.find().limit(3):
                print("\n" + "="*50)
                pprint(doc)
                print("="*50 + "\n")
            
            # Mostra estrutura do primeiro documento
            if count > 0:
                sample = self.collection.find_one()
                logger.info("\n📝 Estrutura do documento:")
                pprint({k: type(v) for k, v in sample.items()})
                
        except Exception as e:
            logger.error(f"Erro ao acessar dados: {str(e)}")

    def close(self):
        """Fecha a conexão"""
        if self.client:
            self.client.close()
            logger.info("Conexão fechada")

if __name__ == "__main__":
    tester = MongoDBTester()
    
    print("\n" + "="*60)
    print("TESTE DE CONEXÃO COM MONGODB".center(60))
    print("="*60 + "\n")
    
    try:
        # Testa a conexão e mostra dados
        tester.show_sample_data()
    finally:
        tester.close()
    
    print("\n" + "="*60)
    print("FIM DO TESTE".center(60))
    print("="*60 + "\n")
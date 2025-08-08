import os
import sys
from pymongo import MongoClient
from gridfs import GridFS
from docx import Document
from bson import ObjectId
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
import time
from datetime import datetime, timedelta
import logging
from typing import Optional, Dict, List, Tuple
import hashlib

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Configurações
class Config:
    MONGO_URI = os.getenv("MONGO_URI", "mongodb://root:root@uabbot-mongodb-1:27017/")
    DB_NAME = os.getenv("DB_NAME", "uab")
    DOCUMENTS_DIR = "/app/data/documents"
    MODEL_CACHE_DIR = "/app/data/model_cache"
    EMBEDDINGS_PATH = "/app/data/embeddings.pkl"
    MAX_WAIT_TIME = 300  # segundos
    RETRY_INTERVAL = 5  # segundos
    MODEL_NAME = "all-MiniLM-L6-v2"  # Modelo padrão
    FALLBACK_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"  # Modelo alternativo
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    OCR_MAX_PAGES = 10
    SCAN_INTERVAL = 60 * 5  # 5 minutos entre verificações (em segundos)
    LAST_SCAN_FILE = "/app/data/last_scan.txt"  # Arquivo para armazenar último horário de verificação

    @classmethod
    def initialize(cls):
        """Configura ambiente e verifica diretórios"""
        os.makedirs(cls.DOCUMENTS_DIR, exist_ok=True)
        os.makedirs(cls.MODEL_CACHE_DIR, exist_ok=True)
        
        # Configuração de cache
        os.environ['HF_HOME'] = cls.MODEL_CACHE_DIR
        os.environ['TRANSFORMERS_CACHE'] = cls.MODEL_CACHE_DIR
        os.environ['TORCH_HOME'] = cls.MODEL_CACHE_DIR

class MongoDBManager:
    """Gerenciador de conexão com MongoDB com recursos avançados"""
    
    def __init__(self):
        self.client = None
        self.db = None
        self.fs = None
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
    
    def connect(self):
        """Estabelece conexão com MongoDB com retry"""
        start_time = time.time()
        last_error = None
        
        while time.time() - start_time < Config.MAX_WAIT_TIME:
            try:
                self.client = MongoClient(
                    Config.MONGO_URI,
                    serverSelectionTimeoutMS=5000,
                    connectTimeoutMS=10000,
                    socketTimeoutMS=10000,
                    retryWrites=True,
                    retryReads=True
                )
                self.client.admin.command('ping')
                self.db = self.client[Config.DB_NAME]
                self.fs = GridFS(self.db)
                logger.info("✅ Conexão com MongoDB estabelecida com sucesso")
                return True
            except Exception as e:
                last_error = e
                logger.warning(f"Aguardando MongoDB... ({str(e)})")
                time.sleep(Config.RETRY_INTERVAL)
        
        logger.error("❌ Não foi possível conectar ao MongoDB após várias tentativas")
        raise ConnectionError(f"Não foi possível conectar ao MongoDB: {str(last_error)}")
    
    def disconnect(self):
        """Fecha conexão com MongoDB"""
        if self.client:
            self.client.close()
            self.client = None
            logger.info("Conexão com MongoDB encerrada")

class DocumentProcessor:
    """Processador de documentos com suporte para múltiplos formatos"""
    
    @staticmethod
    def calculate_file_hash(filepath: str) -> str:
        """Calcula hash SHA-256 do arquivo para verificação de integridade"""
        sha256 = hashlib.sha256()
        with open(filepath, 'rb') as f:
            while chunk := f.read(8192):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    @staticmethod
    def validate_file(filepath: str, filename: str) -> bool:
        """Valida o arquivo antes do processamento"""
        if not os.path.exists(filepath):
            logger.error(f"Arquivo não encontrado: {filename}")
            return False
        
        if os.path.getsize(filepath) > Config.MAX_FILE_SIZE:
            logger.error(f"Arquivo muito grande: {filename} ({os.path.getsize(filepath)/1024/1024:.2f}MB)")
            return False
            
        return True
    
    @classmethod
    def process_pdf(cls, fs: GridFS, db, filepath: str, filename: str) -> Optional[Dict]:
        """Processa arquivo PDF com fallback para múltiplos métodos de extração"""
        if not cls.validate_file(filepath, filename):
            return None
            
        text = ""
        extraction_method = "failed"
        file_hash = cls.calculate_file_hash(filepath)
        
        # Verifica se documento já existe com mesmo hash
        existing = db.documents.find_one({"file_hash": file_hash})
        if existing:
            logger.info(f"⚠️ Documento {filename} já processado anteriormente. Pulando...")
            return None
        
        # Tentativa 1: PyMuPDF (fitz)
        try:
            import fitz
            doc = fitz.open(filepath)
            text = " ".join([page.get_text() for page in doc])
            doc.close()
            extraction_method = "pymupdf"
        except Exception as e:
            logger.warning(f"PyMuPDF não conseguiu extrair texto de {filename}: {str(e)}")
        
        # Tentativa 2: pdfplumber (se PyMuPDF falhou)
        if not text.strip():
            try:
                import pdfplumber
                with pdfplumber.open(filepath) as pdf:
                    text = " ".join([page.extract_text() or "" for page in pdf.pages])
                extraction_method = "pdfplumber"
            except Exception as e:
                logger.warning(f"pdfplumber não conseguiu extrair texto de {filename}: {str(e)}")
        
        # Tentativa 3: OCR (se os métodos anteriores falharam)
        if not text.strip():
            text = cls.extract_text_with_ocr(filepath)
            extraction_method = "ocr" if text.strip() else "failed"
        
        try:
            with open(filepath, 'rb') as file:
                file_id = fs.put(file, filename=filename)
                
                doc_data = {
                    "file_id": file_id,
                    "filename": filename,
                    "type": "pdf",
                    "text_content": text.strip(),
                    "extraction_method": extraction_method,
                    "file_hash": file_hash,
                    "timestamp": datetime.utcnow()
                }
                
                result = db.documents.insert_one(doc_data)
                doc_data["_id"] = result.inserted_id
                
                if text.strip():
                    logger.info(f"✅ PDF {filename} processado com sucesso (método: {extraction_method})")
                else:
                    logger.error(f"❌ Falha ao extrair texto do PDF {filename}")
                
                return doc_data
        except Exception as e:
            logger.error(f"❌ Erro ao armazenar PDF {filename}: {str(e)}")
            return None
    
    @classmethod
    def extract_text_with_ocr(cls, filepath: str) -> str:
        """Extrai texto de PDFs baseados em imagem usando OCR"""
        try:
            import pytesseract
            from pdf2image import convert_from_path
            
            images = convert_from_path(filepath, dpi=300, first_page=0, last_page=Config.OCR_MAX_PAGES)
            text = ""
            
            for img in images:
                text += pytesseract.image_to_string(img) + "\n"
            
            return text.strip()
        except ImportError:
            logger.error("Bibliotecas de OCR não instaladas. Instale pytesseract e pdf2image.")
            return ""
        except Exception as e:
            logger.error(f"Erro no OCR: {str(e)}")
            return ""
    
    @classmethod
    def process_docx(cls, fs: GridFS, db, filepath: str, filename: str) -> Optional[Dict]:
        """Processa arquivo DOCX/DOC"""
        if not cls.validate_file(filepath, filename):
            return None
            
        file_hash = cls.calculate_file_hash(filepath)
        
        # Verifica se documento já existe com mesmo hash
        existing = db.documents.find_one({"file_hash": file_hash})
        if existing:
            logger.info(f"⚠️ Documento {filename} já processado anteriormente. Pulando...")
            return None
        
        try:
            with open(filepath, 'rb') as file:
                doc = Document(file)
                text = " ".join([para.text for para in doc.paragraphs if para.text])
                
                file.seek(0)
                file_id = fs.put(file, filename=filename)
                
                doc_data = {
                    "file_id": file_id,
                    "filename": filename,
                    "type": "docx",
                    "text_content": text,
                    "file_hash": file_hash,
                    "timestamp": datetime.utcnow()
                }
                
                db.documents.insert_one(doc_data)
                logger.info(f"✅ DOCX {filename} carregado com sucesso")
                return doc_data
        except Exception as e:
            logger.error(f"❌ Erro ao processar DOCX {filename}: {str(e)}")
            return None

class EmbeddingManager:
    """Gerenciador de modelos de embedding e operações relacionadas"""
    
    @staticmethod
    def load_model() -> SentenceTransformer:
        """Carrega o modelo SentenceTransformer com fallback"""
        try:
            logger.info(f"⏳ Carregando modelo {Config.MODEL_NAME}...")
            
            model = SentenceTransformer(
                Config.MODEL_NAME,
                cache_folder=Config.MODEL_CACHE_DIR,
                device='cpu'
            )
            
            # Teste de funcionamento
            test_embedding = model.encode("teste")
            if not isinstance(test_embedding, np.ndarray):
                raise ValueError("Modelo não retornou embedding válido")
                
            logger.info(f"✅ Modelo {Config.MODEL_NAME} carregado com sucesso")
            return model
        except Exception as e:
            logger.error(f"❌ Falha ao carregar {Config.MODEL_NAME}: {str(e)}")
            
            if Config.MODEL_NAME != Config.FALLBACK_MODEL:
                logger.info(f"Tentando carregar modelo alternativo {Config.FALLBACK_MODEL}...")
                Config.MODEL_NAME = Config.FALLBACK_MODEL
                return EmbeddingManager.load_model()
            
            raise
    
    @staticmethod
    def create_search_indexes(db):
        """Cria índices de busca otimizados"""
        try:
            existing_indexes = {idx['name']: idx for idx in db.documents.list_indexes()}
            
            # Índice de texto full-text
            if 'text_content_text' not in existing_indexes:
                db.documents.create_index(
                    [("text_content", "text")],
                    name="text_content_text",
                    default_language="portuguese"
                )
            
            # Índice único para filename
            if 'filename_1' not in existing_indexes:
                db.documents.create_index(
                    [("filename", 1)],
                    name="filename_1",
                    unique=True
                )
            
            # Índice para file_hash (evita duplicatas)
            if 'file_hash_1' not in existing_indexes:
                db.documents.create_index(
                    [("file_hash", 1)],
                    name="file_hash_1",
                    unique=True,
                    partialFilterExpression={"file_hash": {"$exists": True}}
                )
            
            logger.info("✅ Índices de busca configurados com sucesso")
        except Exception as e:
            logger.error(f"❌ Erro ao configurar índices: {str(e)}")
            raise
    
    @staticmethod
    def save_embeddings(documents: List[Dict], embeddings: List[np.ndarray]) -> bool:
        """Salva embeddings com compressão"""
        try:
            data = {
                'documents': documents,
                'embeddings': embeddings,
                'model': Config.MODEL_NAME,
                'timestamp': datetime.utcnow()
            }
            
            with open(Config.EMBEDDINGS_PATH, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            logger.info(f"✅ Embeddings salvos para {len(documents)} documentos")
            return True
        except Exception as e:
            logger.error(f"❌ Erro ao salvar embeddings: {str(e)}")
            return False
    
    @staticmethod
    def load_embeddings() -> Tuple[List[Dict], List[np.ndarray]]:
        """Carrega embeddings existentes"""
        try:
            if os.path.exists(Config.EMBEDDINGS_PATH):
                with open(Config.EMBEDDINGS_PATH, 'rb') as f:
                    data = pickle.load(f)
                    return data.get('documents', []), data.get('embeddings', [])
            return [], []
        except Exception as e:
            logger.error(f"❌ Erro ao carregar embeddings: {str(e)}")
            return [], []

class DocumentLoader:
    """Classe principal para carregamento de documentos"""
    
    @classmethod
    def load_documents(cls, initial_load: bool = True) -> bool:
        """Processa documentos no diretório configurado"""
        try:
            with MongoDBManager() as mongo:
                # Configura índices apenas no carregamento inicial
                if initial_load:
                    EmbeddingManager.create_search_indexes(mongo.db)
                
                # Verifica diretório de documentos
                if not os.path.exists(Config.DOCUMENTS_DIR):
                    logger.error(f"Diretório de documentos não encontrado: {Config.DOCUMENTS_DIR}")
                    return False
                
                # Carrega modelo de embeddings
                model = EmbeddingManager.load_model()
                
                # Carrega embeddings existentes
                existing_docs, existing_embeddings = EmbeddingManager.load_embeddings()
                existing_filenames = {doc['filename'] for doc in existing_docs}
                
                documents: List[Dict] = existing_docs.copy()
                embeddings: List[np.ndarray] = existing_embeddings.copy()
                processed_files = 0
                
                for filename in os.listdir(Config.DOCUMENTS_DIR):
                    # Pula arquivos já processados em cargas incrementais
                    if not initial_load and filename in existing_filenames:
                        continue
                        
                    filepath = os.path.join(Config.DOCUMENTS_DIR, filename)
                    
                    try:
                        doc_data = None
                        
                        if filename.lower().endswith('.pdf'):
                            doc_data = DocumentProcessor.process_pdf(mongo.fs, mongo.db, filepath, filename)
                        elif filename.lower().endswith(('.doc', '.docx')):
                            doc_data = DocumentProcessor.process_docx(mongo.fs, mongo.db, filepath, filename)
                        else:
                            logger.warning(f"⚠️ Formato não suportado: {filename}")
                            continue
                        
                        if doc_data and doc_data.get('text_content'):
                            embedding = model.encode(doc_data['text_content'])
                            documents.append({
                                'filename': filename,
                                'text': doc_data['text_content'],
                                'metadata': doc_data
                            })
                            embeddings.append(embedding)
                            processed_files += 1
                            
                    except Exception as e:
                        logger.error(f"❌ Erro ao processar {filename}: {str(e)}", exc_info=True)
                        continue
                
                if processed_files > 0 or initial_load:
                    EmbeddingManager.save_embeddings(documents, embeddings)
                
                logger.info(f"📊 Total de novos documentos processados: {processed_files}")
                return processed_files > 0
                
        except Exception as e:
            logger.error(f"❌ Falha crítica no carregamento de documentos: {str(e)}", exc_info=True)
            return False
    
    @classmethod
    def continuous_loading(cls):
        """Executa o carregamento contínuo de documentos"""
        logger.info("🔄 Iniciando carregamento contínuo de documentos...")
        
        # Primeira carga completa
        cls.load_documents(initial_load=True)
        
        while True:
            try:
                start_time = time.time()
                
                logger.info("⏳ Verificando novos documentos...")
                files_to_process = cls._get_files_to_process()
                
                if files_to_process:
                    logger.info(f"📁 Encontrados {len(files_to_process)} arquivos para processar")
                    if cls.load_documents(initial_load=False):
                        logger.info("✅ Documentos atualizados com sucesso")
                else:
                    logger.info("⏳ Nenhum arquivo novo ou modificado encontrado")
                
                # Calcula tempo de processamento
                processing_time = time.time() - start_time
                sleep_time = max(0, Config.SCAN_INTERVAL - processing_time)
                
                logger.info(f"⏳ Próxima verificação em {sleep_time:.1f} segundos...")
                time.sleep(sleep_time)
                
            except KeyboardInterrupt:
                logger.info("🛑 Interrupção recebida, encerrando...")
                break
            except Exception as e:
                logger.error(f"⚠️ Erro no carregamento contínuo: {str(e)}")
                logger.info("🔄 Tentando novamente em 1 minuto...")
                time.sleep(60)
    
    @classmethod
    def _get_files_to_process(cls) -> List[str]:
        """Retorna lista de arquivos que precisam ser processados"""
        try:
            with MongoDBManager() as mongo:
                # Obtém todos os arquivos no diretório
                current_files = set(os.listdir(Config.DOCUMENTS_DIR))
                
                # Obtém todos os arquivos já processados do MongoDB
                processed_files = {doc['filename'] for doc in mongo.db.documents.find({}, {'filename': 1})}
                
                # Arquivos novos (presentes no diretório mas não no MongoDB)
                new_files = list(current_files - processed_files)
                
                # Verifica também por arquivos modificados (comparando hashes)
                modified_files = []
                for filename in current_files & processed_files:
                    filepath = os.path.join(Config.DOCUMENTS_DIR, filename)
                    if os.path.isfile(filepath):
                        current_hash = DocumentProcessor.calculate_file_hash(filepath)
                        db_doc = mongo.db.documents.find_one({'filename': filename}, {'file_hash': 1})
                        if db_doc and db_doc.get('file_hash') != current_hash:
                            modified_files.append(filename)
                
                return new_files + modified_files
                
        except Exception as e:
            logger.error(f"❌ Erro ao verificar arquivos para processar: {str(e)}")
            return []

    # Adicione esta função ao DocumentLoader
    @classmethod
    def should_process_file(cls, filename: str, db) -> bool:
        """Verifica se um arquivo deve ser processado"""
        filepath = os.path.join(Config.DOCUMENTS_DIR, filename)
        if not os.path.isfile(filepath):
            return False
        
        current_hash = DocumentProcessor.calculate_file_hash(filepath)
        existing_doc = db.documents.find_one({"filename": filename})
        
        # Se o arquivo não existe no banco ou o hash mudou
        return not existing_doc or existing_doc.get("file_hash") != current_hash

    # Modifique o método continuous_loading para:
    @classmethod
    def continuous_loading(cls):
        """Executa o carregamento contínuo de documentos"""
        logger.info("🔄 Iniciando carregamento contínuo de documentos...")
        
        # Primeira carga completa
        cls.load_documents(initial_load=True)
        
        while True:
            try:
                start_time = time.time()
                logger.info(f"⏳ Iniciando verificação em {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                with MongoDBManager() as mongo:
                    # Lista todos os arquivos no diretório
                    current_files = set(os.listdir(Config.DOCUMENTS_DIR))
                    
                    # Processa apenas arquivos que precisam ser atualizados
                    files_to_process = [
                        f for f in current_files 
                        if f.lower().endswith(('.pdf', '.doc', '.docx')) 
                        and cls.should_process_file(f, mongo.db)
                    ]
                    
                    if files_to_process:
                        logger.info(f"📁 Arquivos para processar: {files_to_process}")
                        model = EmbeddingManager.load_model()
                        documents, embeddings = EmbeddingManager.load_embeddings()
                        
                        for filename in files_to_process:
                            filepath = os.path.join(Config.DOCUMENTS_DIR, filename)
                            try:
                                doc_data = None
                                
                                if filename.lower().endswith('.pdf'):
                                    doc_data = DocumentProcessor.process_pdf(mongo.fs, mongo.db, filepath, filename)
                                elif filename.lower().endswith(('.doc', '.docx')):
                                    doc_data = DocumentProcessor.process_docx(mongo.fs, mongo.db, filepath, filename)
                                
                                if doc_data and doc_data.get('text_content'):
                                    embedding = model.encode(doc_data['text_content'])
                                    documents.append({
                                        'filename': filename,
                                        'text': doc_data['text_content'],
                                        'metadata': doc_data
                                    })
                                    embeddings.append(embedding)
                                    logger.info(f"✅ {filename} processado com sucesso")
                            
                            except Exception as e:
                                logger.error(f"❌ Erro ao processar {filename}: {str(e)}")
                                continue
                        
                        if files_to_process:
                            EmbeddingManager.save_embeddings(documents, embeddings)
                            logger.info(f"💾 Embeddings atualizados para {len(files_to_process)} documentos")
                    else:
                        logger.info("⏳ Nenhum arquivo novo ou modificado encontrado")
                
                # Calcula tempo até próxima verificação
                processing_time = time.time() - start_time
                sleep_time = max(0, Config.SCAN_INTERVAL - processing_time)
                
                logger.info(f"⏳ Próxima verificação em {sleep_time:.1f} segundos...")
                time.sleep(sleep_time)
                
            except KeyboardInterrupt:
                logger.info("🛑 Interrupção recebida, encerrando...")
                break
            except Exception as e:
                logger.error(f"⚠️ Erro no carregamento contínuo: {str(e)}")
                logger.info("🔄 Tentando novamente em 1 minuto...")
                time.sleep(60)
                
def main():
    """Função principal"""
    try:
        Config.initialize()
        
        if len(sys.argv) > 1 and sys.argv[1] == "--continuous":
            logger.info("🔁 Modo contínuo ativado")
            DocumentLoader.continuous_loading()
        else:
            logger.info("⚡ Executando carregamento único")
            if DocumentLoader.load_documents(initial_load=True):
                logger.info("✅ Carregamento concluído com sucesso!")
                sys.exit(0)
            else:
                logger.error("❌ Falha no carregamento de documentos")
                sys.exit(1)
                
    except KeyboardInterrupt:
        logger.info("🛑 Programa interrompido pelo usuário")
        sys.exit(0)
    except Exception as e:
        logger.error(f"💥 Falha crítica: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
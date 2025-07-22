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
from datetime import datetime
import logging
from typing import Optional, Dict, List

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Configuração
MONGO_URI = os.getenv("MONGO_URI", "mongodb://root:root@uabbot-mongodb-1:27017/")
DB_NAME = os.getenv("DB_NAME", "uab")
DOCUMENTS_DIR = "/app/data/documents"

# Tempo máximo de espera pelo MongoDB (em segundos)
MAX_WAIT_TIME = 300
RETRY_INTERVAL = 5

class MongoDBConnection:
    """Classe para gerenciar conexão com o MongoDB"""
    def __init__(self):
        self.client = None
        self.db = None
        self.fs = None
    
    def __enter__(self):
        self.client = self._wait_for_mongodb()
        self.db = self.client[DB_NAME]
        self.fs = GridFS(self.db)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            self.client.close()
    
    def _wait_for_mongodb(self) -> MongoClient:
        """Aguarda até que o MongoDB esteja disponível"""
        start_time = time.time()
        
        while time.time() - start_time < MAX_WAIT_TIME:
            try:
                client = MongoClient(
                    MONGO_URI,
                    serverSelectionTimeoutMS=5000,
                    connectTimeoutMS=10000,
                    socketTimeoutMS=10000,
                    retryWrites=True,
                    retryReads=True
                )
                # Verifica se a conexão está ativa
                client.admin.command('ping')
                logger.info("✅ Conexão com MongoDB estabelecida com sucesso")
                return client
            except Exception as e:
                logger.warning(f"Aguardando MongoDB... ({str(e)})")
                time.sleep(RETRY_INTERVAL)
        
        logger.error("❌ Não foi possível conectar ao MongoDB após várias tentativas")
        raise ConnectionError("Não foi possível conectar ao MongoDB")

def store_pdf(fs: GridFS, db, filepath: str, filename: str) -> Optional[Dict]:
    """Processa um arquivo PDF e retorna seus metadados"""
    text = ""
    extraction_method = "failed"
    
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
        text = extract_text_with_ocr(filepath)
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

def extract_text_with_ocr(filepath: str, max_pages: int = 10) -> str:
    """Extrai texto de PDFs baseados em imagem usando OCR"""
    try:
        import pytesseract
        from pdf2image import convert_from_path
        
        images = convert_from_path(filepath, dpi=300)
        text = ""
        
        for i, img in enumerate(images[:max_pages]):
            text += pytesseract.image_to_string(img) + "\n"
        
        return text.strip()
    except ImportError:
        logger.error("Bibliotecas de OCR não instaladas. Instale pytesseract e pdf2image.")
        return ""
    except Exception as e:
        logger.error(f"Erro no OCR: {str(e)}")
        return ""

def store_docx(fs: GridFS, db, filepath: str, filename: str) -> Optional[Dict]:
    """Processa um arquivo DOCX e retorna seus metadados"""
    try:
        if db.documents.find_one({"filename": filename}):
            logger.info(f"⚠️ Documento {filename} já carregado. Pulando...")
            return None

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
                "timestamp": datetime.utcnow()
            }
            
            db.documents.insert_one(doc_data)
            logger.info(f"✅ DOCX {filename} carregado com sucesso")
            return doc_data
    except Exception as e:
        logger.error(f"❌ Erro ao processar DOCX {filename}: {str(e)}")
        return None

def load_documents(model: SentenceTransformer) -> bool:
    """Carrega todos os documentos do diretório para o MongoDB"""
    try:
        with MongoDBConnection() as mongo:
            # Cria índice para evitar duplicatas
            mongo.db.documents.create_index("filename", unique=True)
            
            if not os.path.exists(DOCUMENTS_DIR):
                logger.error(f"Diretório de documentos não encontrado: {DOCUMENTS_DIR}")
                return False
            
            documents: List[Dict] = []
            embeddings: List[np.ndarray] = []
            
            for filename in os.listdir(DOCUMENTS_DIR):
                filepath = os.path.join(DOCUMENTS_DIR, filename)
                
                try:
                    if filename.lower().endswith('.pdf'):
                        doc_data = store_pdf(mongo.fs, mongo.db, filepath, filename)
                    elif filename.lower().endswith(('.doc', '.docx')):
                        doc_data = store_docx(mongo.fs, mongo.db, filepath, filename)
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
                        
                except Exception as e:
                    logger.error(f"❌ Erro ao processar {filename}: {str(e)}")
                    continue
            
            if documents:
                save_embeddings(documents, embeddings)
            
            return True
    except Exception as e:
        logger.error(f"❌ Falha crítica no carregamento de documentos: {str(e)}")
        return False

def save_embeddings(documents: List[Dict], embeddings: List[np.ndarray]) -> bool:
    """Salva os embeddings em um arquivo pickle"""
    try:
        os.makedirs('/app/data', exist_ok=True)
        with open('/app/data/embeddings.pkl', 'wb') as f:
            pickle.dump({
                'documents': documents,
                'embeddings': embeddings,
                'timestamp': datetime.utcnow()
            }, f)
        logger.info(f"✅ Embeddings salvos para {len(documents)} documentos")
        return True
    except Exception as e:
        logger.error(f"❌ Erro ao salvar embeddings: {str(e)}")
        return False

if __name__ == "__main__":
    logger.info("⏳ Iniciando carregamento de documentos...")
    
    try:
        # Inicializa o modelo de embeddings
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        if load_documents(model):
            logger.info("✅ Carregamento de documentos concluído com sucesso!")
            sys.exit(0)
        else:
            logger.error("❌ Falha no carregamento de documentos")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"❌ Falha crítica: {str(e)}")
        sys.exit(1)
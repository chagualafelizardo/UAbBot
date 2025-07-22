import os
import PyPDF2
from pymongo import MongoClient

def extract_text_from_pdf(pdf_path):
    """Extrai texto de um arquivo PDF"""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""  # Evita valores None
        return text

def store_pdf_in_mongodb(pdf_path, mongodb_uri='mongodb://root:root@uabbot-mongodb-1:27017/', db_name='uab'):
    """Armazena o PDF no MongoDB"""
    # Extrai o texto
    pdf_text = extract_text_from_pdf(pdf_path)
    
    # Conexão com o MongoDB
    client = MongoClient(mongodb_uri)
    db = client[db_name]
    collection = db['documents']
    
    # Prepara o documento para armazenamento
    document = {
        'filename': os.path.basename(pdf_path),
        'content': pdf_text,
        'filetype': 'pdf',
        'processed': False,
        'metadata': {
            'pages': len(PyPDF2.PdfReader(pdf_path).pages),
            'size': os.path.getsize(pdf_path)
        }
    }
    
    # Insere no MongoDB
    result = collection.insert_one(document)
    print(f"PDF carregado com sucesso! ID: {result.inserted_id}")
    return result.inserted_id

# USO: Escolha UMA das opções abaixo para o caminho do PDF:

# Opção 1: Usar raw string (recomendado para Windows)
pdf_path = r"C:\Users\Felizardo Chaguala\Desktop\test.pdf"

# Opção 2: Usar barras normais (funciona no Windows também)
# pdf_path = "C:/Users/Felizardo Chaguala/Desktop/test.pdf"

# Opção 3: Usar dupla barra invertida
# pdf_path = "C:\\Users\\Felizardo Chaguala\\Desktop\\test.pdf"

store_pdf_in_mongodb(pdf_path)
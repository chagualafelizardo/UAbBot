import os
from pymongo import MongoClient
from gridfs import GridFS
from PyPDF2 import PdfReader
from docx import Document
from bson import ObjectId

# Configuração
MONGO_URI = os.getenv("MONGO_URI", "mongodb://root:root@uabbot-mongodb-1:27017/")
DB_NAME = os.getenv("DB_NAME", "uab")
DOCUMENTS_DIR = "/app/data/documents"

def load_documents():
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    fs = GridFS(db)
    
    # Garante que a coleção de metadados existe
    if "documents" not in db.list_collection_names():
        db.create_collection("documents")
    
    for filename in os.listdir(DOCUMENTS_DIR):
        filepath = os.path.join(DOCUMENTS_DIR, filename)
        
        if filename.lower().endswith('.pdf'):
            store_pdf(fs, db, filepath, filename)
        elif filename.lower().endswith(('.doc', '.docx')):
            store_docx(fs, db, filepath, filename)
    
    client.close()

def store_pdf(fs, db, filepath, filename):
    try:
        # Tenta extração normal primeiro
        with open(filepath, 'rb') as file:
            reader = PdfReader(file)
            text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
            
            if not text:  # Se não extraiu texto, tenta OCR
                text = extract_text_with_ocr(filepath)
                
            file.seek(0)
            file_id = fs.put(file, filename=filename)
            
            db.documents.insert_one({
                "file_id": file_id,
                "filename": filename,
                "type": "pdf",
                "text_content": text,
                "is_ocr": bool(not text)  # Indica se foi usado OCR
            })
    except Exception as e:
        print(f"Erro ao processar {filename}: {str(e)}")

def extract_text_with_ocr(filepath):
    try:
        import pytesseract
        from pdf2image import convert_from_path
        
        images = convert_from_path(filepath)
        text = ""
        for img in images:
            text += pytesseract.image_to_string(img) + "\n"
        return text
    except ImportError:
        print("Bibliotecas de OCR não instaladas. Instale pytesseract e pdf2image.")
        return ""

def store_docx(fs, db, filepath, filename):
    if db.documents.find_one({"filename": filename}):
        print(f"⚠️  Documento {filename} já carregado. Pulando...")
        return

    with open(filepath, 'rb') as file:
        doc = Document(file)
        text = " ".join([para.text for para in doc.paragraphs])
        
        file.seek(0)
        file_id = fs.put(file, filename=filename)
        
        db.documents.insert_one({
            "file_id": file_id,
            "filename": filename,
            "type": "docx",
            "text_content": text
        })
        print(f"✅ DOCX {filename} carregado.")


if __name__ == "__main__":
    print("Iniciando carregamento de documentos...")
    load_documents()
    print("Documentos carregados com sucesso!")
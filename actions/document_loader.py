import os
from pymongo import MongoClient
from gridfs import GridFS
from PyPDF2 import PdfReader
from docx import Document
from bson import ObjectId
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle

# Configuração
MONGO_URI = os.getenv("MONGO_URI", "mongodb://root:root@uabbot-mongodb-1:27017/")
DB_NAME = os.getenv("DB_NAME", "uab")
DOCUMENTS_DIR = "/app/data/documents"

# Inicialize o modelo de embeddings (adicionar no __main__)
model = SentenceTransformer('all-MiniLM-L6-v2')

def load_documents():
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    fs = GridFS(db)
    
    documents = []
    embeddings = []
    
    for filename in os.listdir(DOCUMENTS_DIR):
        filepath = os.path.join(DOCUMENTS_DIR, filename)
        
        # Verifica se já foi processado
        if db.documents.count_documents({"filename": filename}) > 0:
            continue
            
        if filename.lower().endswith('.pdf'):
            doc_data = store_pdf(fs, db, filepath, filename)
        elif filename.lower().endswith(('.doc', '.docx')):
            doc_data = store_docx(fs, db, filepath, filename)
        
        if doc_data and doc_data.get('text_content'):
            # Gera embedding para o documento
            embedding = model.encode(doc_data['text_content'])
            documents.append({
                'filename': filename,
                'text': doc_data['text_content'],
                'metadata': doc_data
            })
            embeddings.append(embedding)
    
    # Salva os embeddings para uso posterior
    if documents:
        with open('/app/data/embeddings.pkl', 'wb') as f:
            pickle.dump({'documents': documents, 'embeddings': embeddings}, f)
    
    client.close()

def store_pdf(fs, db, filepath, filename):
    # Verifica se já existe um documento com este nome
    if db.documents.find_one({"filename": filename}):
        print(f"⚠️  Documento {filename} já carregado. Pulando...")
        return

    with open(filepath, 'rb') as file:
        reader = PdfReader(file)
        text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
        
        file.seek(0)
        file_id = fs.put(file, filename=filename)
        
        db.documents.insert_one({
            "file_id": file_id,
            "filename": filename,
            "type": "pdf",
            "text_content": text
        })
        print(f"✅ PDF {filename} carregado.")

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
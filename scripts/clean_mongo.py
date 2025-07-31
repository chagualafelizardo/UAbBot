from pymongo import MongoClient
import os

def clean_mongo_indexes():
    client = MongoClient(os.getenv("MONGO_URI", "mongodb://root:root@uabbot-mongodb-1:27017/"))
    db = client[os.getenv("DB_NAME", "uab")]
    
    # Lista e remove todos os índices exceto o _id_
    for index in db.documents.list_indexes():
        if index['name'] != '_id_':
            db.documents.drop_index(index['name'])
    
    client.close()

if __name__ == "__main__":
    clean_mongo_indexes()
    print("Índices do MongoDB limpos com sucesso")
#!/usr/bin/env python3
import sys
import pymongo
import os
from time import sleep

MONGO_URI = os.getenv("MONGO_URI", "mongodb://root:root@uabbot-mongodb-1:27017/")
MAX_RETRIES = 30
RETRY_DELAY = 5

def check_mongo_connection():
    try:
        client = pymongo.MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        client.server_info()  # Testa a conexão
        client.close()
        return True
    except Exception as e:
        print(f"Erro ao conectar ao MongoDB: {str(e)}", file=sys.stderr)
        return False

if __name__ == "__main__":
    for _ in range(MAX_RETRIES):
        if check_mongo_connection():
            sys.exit(0)
        sleep(RETRY_DELAY)
    
    print("Não foi possível conectar ao MongoDB após várias tentativas", file=sys.stderr)
    sys.exit(1)
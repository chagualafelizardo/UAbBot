###### UAbBot: RASA FrameWork ##########
# Projecto desenvolvido por Felizardo L. A. Chaguala, por a obtecao do grau de mestre e Engenharia Informatica e tecnologia Web#

# ########################################################################
https://rasa.com/docs/pro/installation/docker/
# Para criar um projecto novo no RASA usando o docker
docker run -it --rm -v ${PWD}:/app rasa/rasa init --no-prompt
docker run -it --rm -v ${PWD}:/app rasa/rasa train; docker run -it --rm -v ${PWD}:/app rasa/rasa shell nlu

docker run -it --rm -v ${PWD}:/app rasa/rasa shell nlu

# Aqui vou corer o servidor principal rasa
docker run -it --rm -v ${PWD}:/app  -p 5005:5005  rasa/rasa:latest run --enable-api --cors "*" –debug

# O mesmo mais simplificado
docker run -it --rm -v ${PWD}:/app -p 5005:5005 rasa/rasa run --enable-api --cors "*" –debug

# Aqui vou corer o servidor de accoes
docker run -it --rm -v ${PWD}/actions:/app/actions -p 5055:5055 rasa-sdk-custom

# Depois inicie o servidor de ações
docker run -it --rm -v ${PWD}:/app rasa/rasa run actions

# Em outro terminal, inicie o bot
docker run -it --rm -v ${PWD}:/app -p 5005:5005 rasa/rasa shell


# um biblioteca que deve ser adicionada
pip install fuzzywuzzy python-Levenshtein


# Para correr o UAbBot junto de todos os servicos e container basta
docker-compose down; docker-compose up -d --build; docker-compose logs -f
docker-compose build --no-cache
docker-compose up --build 
docker-compose up -d

# Para treinar o meu modelo
docker-compose run rasa rasa train
docker run -it -v ${PWD}:/app rasa/rasa:latest train

# Para correr o modelo e testar o bot
docker-compose run rasa rasa shell
docker run -it -v ${PWD}:/app rasa/rasa:latest shell
docker run -it -v ${PWD}:/app -p 5002:5002 rasa/rasa-x:latest

# Para correr o modelo e verificar a intencao de fala que esta sendo executada
docker-compose run rasa rasa shell nlu

# Quando tenho erro na atualizacao dos arquivos no github
git push -u origin main –force 

# Configuracao para acessar a base de dados do rasa no ambiente grafico
Todos os passos sao semelhantes, apenas este onde no Host name/address deves colocar este parametro {rasa_postgres}

PGADMIN_DEFAULT_EMAIL: pgadmin4@pgadmin.org
PGADMIN_DEFAULT_PASSWORD: postgres

# Documentacao de alguns bugs
3. Problema no rasa_db_setup (que cai após execução)
Isso é esperado - o db_setup deve finalizar após popular o banco. Para mantê-lo ativo:

# TOBE
# 27/05/2025
    Adicionar novas funcionalidades, como integração com APIs ou banco de dados?
    Melhorar o NLP (Processamento de Linguagem Natural) para entender melhor os usuários?
    Testar e depurar algum fluxo de conversa específico?

# 28/05/2025
    Definir intenções e entidades.
    Treinar um modelo (Rasa, LLM, etc.).
    Implementar o backend do chatbot.
    Conectar a um canal (Telegram, Web, WhatsApp, etc.).
    Integrar com uma base de dados ou sistema externo.
    Criar diálogos e fluxos de conversação.
    Usar RAG ou modelos como RoBERTa, GPT, etc.

# https://gapae.uab.pt/perguntas-frequentes/ 
# https://portal.uab.pt/dsd/faqs-2/

# ==============================================
# Para acessar a minha base de dados em MongoDB
docker exec -it uabbot-mongodb-1 mongosh -u root -p root

Valores Válidos para topico (conforme o erro):

Os únicos valores permitidos são:

    "Admissão"

    "Matrículas"

    "Propinas"

    "Currículo"

    "Certificação"

    "Apoio Académico"
    "Recursos Acadêmicos"
    "Ciência e Tecnologia"

    "Outros"

# Para adicionar um novo enum
<!-- db.runCommand({
  collMod: "faqs",
  validator: {
    $jsonSchema: {
      bsonType: "object",
      required: ["pergunta", "resposta", "topico", "data_criacao"],
      properties: {
        topico: {
          bsonType: "string",
          enum: [
            "Admissão",
            "Matrículas",
            "Propinas",
            "Currículo",
            "Certificação",
            "Apoio Académico",
            "Recursos Acadêmicos",
            "Ciência e Tecnologia",
            "Publicações Científicas",
            "Tecnologia",
            "Direitos de Autor",
            "Outros"
          ],
          description: "Categoria temática da FAQ"
        },
        // ... (outras propriedades do validator existente)
      }
    }
  }
}); -->

db.createCollection("faqs", {
  validator: {
    $jsonSchema: {
      bsonType: "object",
      required: ["pergunta", "resposta", "topico", "data_criacao"],
      properties: {
        pergunta: {
          bsonType: "string",
          description: "Texto completo da pergunta frequente"
        },
        resposta: {
          bsonType: "string",
          description: "Resposta detalhada à pergunta"
        },
        topico: {
          bsonType: "string",
          enum: [
            "Admissão",
            "Matrículas",
            "Propinas",
            "Currículo",
            "Certificação",
            "Apoio Académico",
            "Recursos Acadêmicos",
            "Ciência e Tecnologia",
            "Publicações Científicas",
            "Tecnologia",
            "Outros"  // Novo valor adicionado
          ],
          description: "Categoria temática da FAQ"
        },
        cursos_relacionados: {
          bsonType: "array",
          items: {
            bsonType: "string"
          },
          description: "Lista de cursos relacionados à pergunta"
        },
        data_criacao: {
          bsonType: "date",
          description: "Data de criação do registro"
        },
        atualizacao: {
          bsonType: "date",
          description: "Data da última atualização"
        },
        relevancia: {
          bsonType: "int",
          minimum: 1,
          maximum: 5,
          description: "Nível de relevância (1-5)"
        }
      }
    }
  }
});


# ==============================================

Monitoramento do consumo dos recuros no docker
watch -n 5 "docker stats --no-stream"

4. Limpar recursos inúteis (não derruba nada)
Use este comando para limpar redes, volumes e imagens não usadas:

docker system prune -f

mysqldump -u root -p --skip-lock-tables cictra --result-file="C:\Users\Felizardo Chaguala\Documents\OpenMRS\backup_hospital_militar_maputo_30-06-2025.sql"


# =============================================================================================
✅ Por que aparecem fs.files e fs.chunks?

Essas duas coleções são automáticas quando você usa o GridFS com o pymongo/GridFS.

    fs.files: armazena os metadados de cada arquivo (nome, tamanho, tipo, etc.)

    fs.chunks: armazena o conteúdo binário real do arquivo, dividido em blocos

Essas coleções são parte do funcionamento normal do GridFS. Mesmo que você não as mencione diretamente no seu código, o fs.put() do GridFS cuida disso automaticamente.
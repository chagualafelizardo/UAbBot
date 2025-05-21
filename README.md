###### UAbBot: RASA FrameWork ##########
# Projecto desenvolvido por Felizardo L. A. Chaguala, por a obtecao do grau de mestre e Engenharia Informatica e tecnologia Web#

# ########################################################################
https://rasa.com/docs/pro/installation/docker/
# Para criar um projecto novo no RASA usando o docker
docker run -it --rm -v ${PWD}:/app rasa/rasa init --no-prompt
docker run -it --rm -v ${PWD}:/app rasa/rasa train
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
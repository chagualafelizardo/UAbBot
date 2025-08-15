from rasa_sdk import Action
from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
import pymongo
import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from transformers import RobertaTokenizer, RobertaModel
import torch
from datetime import datetime
from collections import defaultdict
import string
import nltk


logger = logging.getLogger(__name__)

class ActionSmartSearch(Action):
    def name(self):
        return "action_search_documents"

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.min_similarity = 0.4
        self.faq_threshold = 0.82
        self.top_k = 5  # Aumentado para capturar mais resultados
        self.context_window = 3
        self.financial_threshold = 0.75  # Limiar específico para questões financeiras
        
        # Conexão com MongoDB
        self.client = pymongo.MongoClient(
            "mongodb://root:root@uabbot-mongodb-1:27017/",
            serverSelectionTimeoutMS=5000,
            socketTimeoutMS=30000,
            connectTimeoutMS=30000,
            retryWrites=True
        )
        self.db = self.client["uab"]
        self.collection = self.db["documents"]
        
        # Modelos de embeddings
        try:
            self.sentence_model = SentenceTransformer(
                'paraphrase-multilingual-MiniLM-L12-v2',
                device='cpu'
            )
            self.roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
            self.roberta_model = RobertaModel.from_pretrained('roberta-base').eval()
        except Exception as e:
            self.logger.error(f"Erro ao carregar modelos: {str(e)}")
            raise

        # Configurações RAG
        self._initialize_rag_environment()

    def _initialize_rag_environment(self):
        """Inicializa o ambiente RAG com índices e coleções necessárias"""
        try:
            # Verificar conexão
            self.client.admin.command('ping')
            
            # Criar coleção para embeddings se não existir
            if "document_embeddings" not in self.db.list_collection_names():
                self.db.create_collection("document_embeddings")
                self.logger.info("Coleção document_embeddings criada")
            
            # Criar índices de texto
            if "text_content_text" not in self.collection.index_information():
                self.collection.create_index(
                    [("text_content", "text")],
                    default_language="portuguese"
                )
            
            # Verificar e completar embeddings ausentes
            self._verify_and_complete_embeddings()
            
        except Exception as e:
            self.logger.error(f"Erro na inicialização do RAG: {str(e)}")
            raise

    def _verify_and_complete_embeddings(self):
        """Verifica e completa embeddings ausentes"""
        try:
            total_docs = self.collection.count_documents({"text_content": {"$exists": True, "$ne": ""}})
            total_embeddings = self.db.document_embeddings.count_documents({})
            
            self.logger.info(f"Documentos: {total_docs}, Embeddings: {total_embeddings}")
            
            if total_docs > total_embeddings:
                self.logger.warning("Alguns documentos estão sem embeddings. Gerando...")
                self._generate_missing_embeddings()
        except Exception as e:
            self.logger.error(f"Erro ao verificar embeddings: {str(e)}")

    def _generate_missing_embeddings(self):
        """Gera embeddings para documentos ausentes"""
        try:
            missing_docs = list(self.collection.aggregate([
                {
                    "$lookup": {
                        "from": "document_embeddings",
                        "localField": "_id",
                        "foreignField": "doc_id",
                        "as": "embeddings"
                    }
                },
                {
                    "$match": {
                        "embeddings": {"$size": 0},
                        "text_content": {"$exists": True, "$ne": ""}
                    }
                },
                {
                    "$limit": 100  # Limite para não sobrecarregar
                }
            ]))

            if not missing_docs:
                self.logger.info("Todos os documentos possuem embeddings")
                return

            self.logger.info(f"Encontrados {len(missing_docs)} documentos sem embeddings")

            for doc in missing_docs:
                try:
                    content = doc.get("text_content", "")
                    if not content:
                        continue

                    embedding = self._get_roberta_embedding(content)
                    
                    self.db.document_embeddings.insert_one({
                        "doc_id": doc["_id"],
                        "filename": doc.get("filename", ""),
                        "embedding": embedding.tolist(),
                        "last_updated": datetime.utcnow(),
                        "content_preview": content[:200] + "..." if len(content) > 200 else content
                    })
                    
                except Exception as e:
                    self.logger.error(f"Erro ao gerar embedding para {doc.get('filename')}: {str(e)}")
                    continue

        except Exception as e:
            self.logger.error(f"Falha ao gerar embeddings ausentes: {str(e)}")

    def _get_roberta_embedding(self, text: str) -> np.ndarray:
        """Gera embeddings com normalização"""
        try:
            inputs = self.roberta_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding='max_length'
            )
            with torch.no_grad():
                outputs = self.roberta_model(**inputs)
            embedding = torch.mean(outputs.last_hidden_state, dim=1).squeeze().numpy()
            return embedding / np.linalg.norm(embedding)
        except Exception as e:
            self.logger.error(f"Erro ao gerar embedding: {str(e)}")
            raise

    def _semantic_search_rag(self, query: str, top_k: int = 5) -> List[Dict]:
        """Busca semântica otimizada com pré-filtro por texto"""
        try:
            # Pré-filtro por texto para melhor performance
            text_ids = [doc['_id'] for doc in self.collection.find(
                {"$text": {"$search": query}},
                {"_id": 1}
            ).limit(100)]
            
            query_embedding = self._get_roberta_embedding(query)
            
            pipeline = [
                {"$match": {"doc_id": {"$in": text_ids}}} if text_ids else {"$match": {}},
                {
                    "$addFields": {
                        "similarity": {
                            "$let": {
                                "vars": {
                                    "dot_product": {
                                        "$reduce": {
                                            "input": {"$zip": {"inputs": ["$embedding", query_embedding.tolist()]}},
                                            "initialValue": 0,
                                            "in": {
                                                "$add": [
                                                    "$$value",
                                                    {"$multiply": [
                                                        {"$arrayElemAt": ["$$this", 0]},
                                                        {"$arrayElemAt": ["$$this", 1]}
                                                    ]}
                                                ]
                                            }
                                        }
                                    },
                                    "query_norm": {
                                        "$sqrt": {
                                            "$reduce": {
                                                "input": query_embedding.tolist(),
                                                "initialValue": 0,
                                                "in": {"$add": ["$$value", {"$pow": ["$$this", 2]}]}
                                            }
                                        }
                                    },
                                    "embedding_norm": {
                                        "$sqrt": {
                                            "$reduce": {
                                                "input": "$embedding",
                                                "initialValue": 0,
                                                "in": {"$add": ["$$value", {"$pow": ["$$this", 2]}]}
                                            }
                                        }
                                    }
                                },
                                "in": {
                                    "$divide": [
                                        "$$dot_product",
                                        {"$multiply": ["$$query_norm", "$$embedding_norm"]}
                                    ]
                                }
                            }
                        }
                    }
                },
                {"$sort": {"similarity": -1}},
                {"$limit": top_k},
                {
                    "$lookup": {
                        "from": "documents",
                        "localField": "doc_id",
                        "foreignField": "_id",
                        "as": "document"
                    }
                },
                {"$unwind": "$document"},
                {
                    "$project": {
                        "doc_id": 1,
                        "similarity": 1,
                        "filename": "$document.filename",
                        "content": "$document.text_content",
                        "content_preview": 1
                    }
                }
            ]
            
            return list(self.db.document_embeddings.aggregate(pipeline))
            
        except Exception as e:
            self.logger.error(f"Erro na busca semântica: {str(e)}", exc_info=True)
            return []

    def _preprocess_text(self, text: str) -> str:
        """Pré-processamento melhorado mantendo pontuação relevante"""
        text = text.lower()
        text = re.sub(r'[^\w\s.,!?]', '', text)  # Mantém pontuação básica
        return text.strip()

    def _is_financial_query(self, query: str) -> bool:
        """Identifica se a consulta é sobre questões financeiras"""
        financial_keywords = [
            "propina", "propinas", "pagamento", "pagar", "financiamento",
            "bolsa", "auxílio", "parcelamento", "dívida", "divida", "valor",
            "preço", "preco", "custos", "isenção", "isencao", "mensalidade"
        ]
        return any(keyword in self._preprocess_text(query) for keyword in financial_keywords)

    def _extract_financial_info(self, content: str) -> Dict:
        """Extrai informações financeiras específicas do conteúdo"""
        financial_data = {
            "payment_options": [],
            "scholarships": [],
            "exemptions": [],
            "contacts": []
        }
        
        # Extrai opções de pagamento
        payment_section = re.search(r'(Opções de Pagamento|Formas de Pagamento)[\s\S]*?(?=\n\s*\n)', content, re.IGNORECASE)
        if payment_section:
            payments = re.findall(r'•\s*([^\n]+)|-\s*([^\n]+)', payment_section.group(0))
            financial_data["payment_options"] = [p[0] or p[1] for p in payments if p[0] or p[1]]
        
        # Extrai bolsas e auxílios
        scholarship_section = re.search(r'(Bolsas|Auxílios|Financiamentos)[\s\S]*?(?=\n\s*\n)', content, re.IGNORECASE)
        if scholarship_section:
            scholarships = re.findall(r'•\s*([^\n]+)|-\s*([^\n]+)', scholarship_section.group(0))
            financial_data["scholarships"] = [s[0] or s[1] for s in scholarships if s[0] or s[1]]
        
        # Extrai contatos
        contact_section = re.search(r'(Contatos|Contactos)[\s\S]*?(?=\n\s*\n)', content, re.IGNORECASE)
        if contact_section:
            contacts = re.findall(r'•\s*([^\n]+)|-\s*([^\n]+)', contact_section.group(0))
            financial_data["contacts"] = [c[0] or c[1] for c in contacts if c[0] or c[1]]
        
        return financial_data

    def _extract_faqs(self, content: str) -> List[Dict]:
        """Extrai FAQs formatadas do conteúdo com separação precisa por perguntas numeradas"""
        faqs = []
        
        # Divide o conteúdo em blocos de perguntas/respostas numeradas
        # Padrão para capturar: número, pergunta e resposta até a próxima pergunta numerada
        faq_blocks = re.findall(
            r'(?:^|\n)(\d+\.)\s*([^\n?]+\??)\s*([^\n]+(?:\n(?!\d+\.\s)[^\n]*)*)', 
            content, 
            re.MULTILINE
        )
        
        for block in faq_blocks:
            number = block[0].strip()
            question = block[1].strip()
            answer = block[2].strip() if block[2] else ""
            
            # Limpa a resposta removendo espaços extras
            answer = re.sub(r'\s+', ' ', answer).strip()
            
            # Verifica se temos conteúdo válido
            if len(question.split()) >= 3 and len(answer.split()) >= 5:
                faqs.append({
                    'number': number,
                    'question': question,
                    'answer': answer,
                    'full_text': f"{number} {question}\n{answer}",
                    'is_financial': any(fin_word in question.lower() for fin_word in ['propina', 'pagamento', 'bolsa'])
                })
        
        return faqs

    def _is_faq_query(self, query: str) -> bool:
        """Determina se a consulta parece uma pergunta de FAQ"""
        question_words = [
            "como", "qual", "quando", "onde", "por que", "quais", "quanto", 
            "contato", "telefone", "email", "endereço", "pode", "deve", "existe",
            "posso", "preciso", "dificuldade", "problema", "ajuda", "dúvida",
            "propina", "propinas", "pagamento", "pagar", "financiamento", "bolsa",
            "conciliar", "conciliação", "equilibrar", "vida profissional", "tempo"
        ]
        
        query_lower = self._preprocess_text(query)
        
        return (
            any(word in query_lower for word in question_words) or
            "?" in query or
            any(phrase in query_lower for phrase in [" o que ", " em que ", " para que "]) or
            re.search(r'\b(pode|deve|como)\s+[^\s]+\s+', query_lower) is not None
        )

    def _find_best_faq_match(self, query: str, docs: List[Dict]) -> Optional[Dict]:
        """Encontra a melhor correspondência de FAQ nos documentos com foco em perguntas numeradas"""
        query_embedding = self.sentence_model.encode(query)
        best_match = None
        highest_score = 0
        
        for doc in docs:
            content = doc.get("content", doc.get("text_content", ""))
            if not content:
                continue
                
            faqs = self._extract_faqs(content)
            if not faqs:
                continue
                
            for faq in faqs:
                # Para perguntas numeradas, verifica correspondência exata com o número
                if re.match(r'^\d+\.', faq['question']):
                    # Extrai o número da pergunta (ex: "5.")
                    question_number = faq['number']
                    # Verifica se o número está na query (ex: query contém "5.")
                    if question_number in query:
                        return {
                            'type': 'faq',
                            'question': faq['question'],
                            'answer': faq['answer'],
                            'score': 1.0,  # Máxima similaridade
                            'filename': doc.get("filename", "documento"),
                            'content': content,
                            'is_financial': faq.get('is_financial', False)
                        }
                    
                # Caso contrário, usa similaridade semântica
                clean_question = self._preprocess_text(faq['question'])
                clean_answer = self._preprocess_text(faq['answer'])
                clean_query = self._preprocess_text(query)
                
                question_emb = self.sentence_model.encode(clean_question)
                answer_emb = self.sentence_model.encode(clean_answer)
                
                question_sim = cosine_similarity([query_embedding], [question_emb])[0][0]
                answer_sim = cosine_similarity([query_embedding], [answer_emb])[0][0]
                similarity = max(question_sim, answer_sim)
                
                # Bônus para FAQs financeiras se a pergunta for financeira
                if self._is_financial_query(query) and faq.get('is_financial', False):
                    similarity = min(similarity + 0.15, 1.0)
                    
                if similarity > highest_score and similarity >= (self.faq_threshold if not self._is_financial_query(query) else self.financial_threshold):
                    highest_score = similarity
                    best_match = {
                        'type': 'faq',
                        'question': faq['question'],
                        'answer': faq['answer'],
                        'score': similarity,
                        'filename': doc.get("filename", "documento"),
                        'content': content,
                        'is_financial': faq.get('is_financial', False)
                    }
        
        return best_match

    def _find_relevant_section(self, content: str, query: str) -> Dict:
        """Encontra a seção mais relevante no conteúdo"""
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        best_paragraph = ""
        best_score = 0
        
        query_embedding = self.sentence_model.encode(query)
        is_financial = self._is_financial_query(query)
        
        for para in paragraphs:
            if len(para.split()) < 5:
                continue
                
            para_embedding = self.sentence_model.encode(para)
            similarity = cosine_similarity([query_embedding], [para_embedding])[0][0]
            
            # Bônus para parágrafos financeiros se a pergunta for financeira
            if is_financial and any(fin_word in para.lower() for fin_word in ['propina', 'pagamento', 'bolsa']):
                similarity = min(similarity + 0.15, 1.0)
            
            if similarity > best_score:
                best_score = similarity
                best_paragraph = para
        
        if best_score > 0.5:
            return {'text': best_paragraph, 'score': best_score}
        
        # Fallback: retorna o início do conteúdo
        return {'text': content[:500], 'score': 0.4}

    def _format_financial_response(self, result: Dict) -> List[Dict]:
        """Formata resposta especializada para questões financeiras"""
        financial_info = self._extract_financial_info(result['content'])
        
        response_parts = [
            {
                'text': f"💳 **Informações sobre pagamentos ({result['filename']}):**",
                'metadata': {"type_speed": 20}
            }
        ]
        
        # Adiciona opções de pagamento
        if financial_info["payment_options"]:
            response_parts.append({
                'text': "\n📋 **Opções de pagamento disponíveis:**\n• " + "\n• ".join(financial_info["payment_options"][:5]),
                'metadata': {"type_speed": 15}
            })
        
        # Adiciona bolsas e auxílios
        if financial_info["scholarships"]:
            response_parts.append({
                'text': "\n🎓 **Bolsas e auxílios:**\n• " + "\n• ".join(financial_info["scholarships"][:3]),
                'metadata': {"type_speed": 15}
            })
        
        # Adiciona contatos
        if financial_info["contacts"]:
            response_parts.append({
                'text': "\n📞 **Contatos úteis:**\n• " + "\n• ".join(financial_info["contacts"][:3]),
                'metadata': {"type_speed": 15}
            })
        
        # Rodapé com ações
        response_parts.append({
            'text': "\n🔍 Para mais detalhes ou solicitar condições especiais, entre em contato com o serviço financeiro.",
            'metadata': {
                'type_speed': 30,
                'buttons': [
                    {
                        'title': '📞 Contato Financeiro',
                        'payload': '/contato_financeiro'
                    },
                    {
                        'title': '📄 Regulamento Completo',
                        'payload': '/regulamento_propinas'
                    }
                ]
            }
        })
        
        return response_parts

    def _generate_related_questions(self, faq_match: Dict) -> List[str]:
        """Gera perguntas relevantes baseadas em análise semântica do conteúdo"""
        try:
            answer = faq_match['answer']
            question = faq_match['question']
            
            # Primeiro, identificamos os tópicos principais na resposta
            main_topics = self._extract_main_topics(answer)
            
            # Depois identificamos conceitos específicos
            specific_concepts = self._extract_specific_concepts(answer)
            
            # Finalmente detectamos ações/recomendações
            actions_recommendations = self._extract_actions_recommendations(answer)
            
            # Geração de perguntas inteligentes
            generated_questions = []
            
            # 1. Perguntas sobre tópicos principais
            for topic in main_topics[:2]:
                if len(topic.split()) <= 4:  # Evita tópicos muito longos
                    generated_questions.append(f"Como {topic} afeta meu desempenho acadêmico?")
                    generated_questions.append(f"Quais são as melhores estratégias para lidar com {topic}?")
            
            # 2. Perguntas sobre conceitos específicos
            for concept in specific_concepts:
                if concept.lower() not in question.lower():  # Evita repetição
                    generated_questions.append(f"Como funciona {concept} na UAb?")
                    generated_questions.append(f"Quem pode me ajudar com questões sobre {concept}?")
            
            # 3. Perguntas sobre ações/recomendações
            for action in actions_recommendations:
                if "inscri" in action:
                    generated_questions.append("Como faço para me inscrever nessa opção?")
                elif "contact" in action or "falar" in action:
                    generated_questions.append("Qual é o contato para essa assistência?")
                elif "estratégia" in action or "método" in action:
                    generated_questions.append("Onde posso aprender mais sobre essas estratégias?")
            
            # Filtro de qualidade
            final_questions = []
            seen = set()
            for q in generated_questions:
                clean_q = q.lower().replace("?", "").strip()
                if (clean_q not in seen and 
                    len(q.split()) > 5 and 
                    not any(word in q.lower() for word in ["qual é a função", "como o"])):
                    seen.add(clean_q)
                    final_questions.append(q)
            
            return final_questions[:3] if final_questions else []

        except Exception as e:
            self.logger.error(f"Erro na geração de perguntas: {str(e)}")
            return []

    def _extract_main_topics(self, text: str) -> List[str]:
        """Identifica os tópicos principais mencionados no texto"""
        # Padrão para frases importantes (normalmente após dois pontos ou marcadores)
        topics = re.findall(r'(?:^|\n|•\s)([A-Z][^.:?!]+?)(?=[.:?!]|\n|$)', text)
        
        # Filtra tópicos por relevância
        filtered = []
        common_words = {"isso", "que", "qual", "como", "quando"}
        for topic in topics:
            words = topic.split()
            if (4 <= len(words) <= 8 and 
                not any(w.lower() in common_words for w in words[:3])):
                filtered.append(topic.strip())
        
        return list(dict.fromkeys(filtered))[:5]  # Remove duplicados e limita

    def _extract_specific_concepts(self, text: str) -> List[str]:
        """Extrai conceitos específicos (siglas, nomes próprios, termos técnicos)"""
        # Padrão para siglas (2-4 letras maiúsculas)
        acronyms = re.findall(r'\b[A-Z]{2,4}\b', text)
        
        # Padrão para termos técnicos entre aspas ou em itálico
        terms = re.findall(r'"(.*?)"|\'(.*?)\'|\b([A-Z][a-z]+ [A-Z][a-z]+)\b', text)
        flat_terms = [t for group in terms for t in group if t]
        
        # Padrão para nomes de serviços/departamentos
        services = re.findall(r'\b([A-Z][a-z]+ (?:de|do|da) [A-Z][a-z]+)\b', text)
        
        # Combina todos os conceitos
        all_concepts = acronyms + flat_terms + services
        return list(dict.fromkeys(all_concepts))[:5]  # Remove duplicados e limita

    def _extract_actions_recommendations(self, text: str) -> List[str]:
        """Identifica ações ou recomendações específicas no texto"""
        # Padrão para verbos no infinitivo seguidos de complemento
        actions = re.findall(r'\b([Pp]ode [a-záéíóúâêôçãõ]+)\b|\b([Dd]eve [a-záéíóúâêôçãõ]+)\b', text)
        flat_actions = [a for group in actions for a in group if a]
        
        # Padrão para recomendações explícitas
        recommendations = re.findall(r'\b([Rr]ecomenda-se [^.!?]+)|([Cc]ontate [^.!?]+)', text)
        flat_recommendations = [r for group in recommendations for r in group if r]
        
        return (flat_actions + flat_recommendations)[:5]  # Limita a 5 itens

    def _extract_context_keywords(self, text: str) -> Dict[str, List[str]]:
        """Extrai palavras-chave contextuais sem usar NLTK"""
        keywords = {
            "prazo": [],
            "documento": [],
            "contato": [],
            "requisito": []
        }
        
        # Mapeamento de padrões
        patterns = {
            "prazo": r"(prazo|até|dia|data|vencimento).{0,20}?\b(\d{1,2}\/\d{1,2}\/\d{2,4}|\d+\s+dias?)\b",
            "documento": r"(documento|formulário|requerimento|comprovante).{0,20}?(entregar|apresentar|necessário)",
            "contato": r"(contatar|telefone|email|setor|secretaria).{0,20}?\b(\+?\d{9,15}|\b[\w\.-]+@[\w\.-]+\.\w{2,}\b)",
            "requisito": r"(requer|necessário|precisa).{0,15}?(ter|possuir|estar)"
        }
        
        for key, pattern in patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if match.group():
                    keywords[key].append(match.group())
        
        return keywords

    def _extract_entities(self, text: str) -> List[str]:
        """Extrai entidades importantes sem usar NLTK"""
        # Padrões para entidades (maiúsculas após pontuação ou no início)
        entities = re.findall(r"(?:^|[.!?]\s+)([A-Z][a-záéíóúâêôçãõ]+(?:\s+[A-Z][a-záéíóúâêôçãõ]+)*)", text)
        
        # Filtra entidades comuns e repetidas
        common_entities = {"Universidade", "Aluno", "Curso", "Processo"}
        unique_entities = []
        seen = set()
        
        for entity in entities:
            clean_entity = entity.strip()
            if (clean_entity not in seen and 
                clean_entity not in common_entities and
                len(clean_entity.split()) < 4):
                seen.add(clean_entity)
                unique_entities.append(clean_entity)
        
        return unique_entities[:5]  # Limita a 5 entidades

    def _extract_actions(self, text: str) -> List[Tuple[str, str]]:
        """Extrai pares ação-alvo sem usar NLTK"""
        # Padrão: verbo + substantivo
        actions = []
        matches = re.finditer(r"\b([a-záéíóúâêôçãõ]+r)\b.{0,10}?\b([a-záéíóúâêôçãõ]{3,}s?\b)", text, re.IGNORECASE)
        
        for match in matches:
            verb = match.group(1).lower()
            noun = match.group(2).lower()
            
            # Filtra verbos comuns
            if verb not in {"ser", "ter", "haver", "poder"}:
                actions.append((verb, noun))
        
        return actions[:5]  # Limita a 5 ações

    def _format_faq_response(self, faq_match: Dict) -> List[Dict]:
        """Formata resposta para FAQ com perguntas contextualizadas"""
        response_parts = [
            {
                'text': f"❓ **Pergunta encontrada em {faq_match['filename']}:**\n{faq_match['question']}",
                'metadata': {
                    'response_part': 'question',
                    'complete_before_next': True
                }
            },
            {
                'text': f"✅ **Resposta completa:**\n{faq_match['answer']}",
                'metadata': {
                    'response_part': 'answer',
                    'complete_before_next': True
                }
            }
        ]
        
        # Gera perguntas contextualizadas
        related_questions = self._generate_related_questions(faq_match)
        
        if related_questions:
            questions_text = "\n\n🔍 Com base nesta informação, você pode querer saber:"
            response_parts.append({
                'text': questions_text,
                'metadata': {
                    'type_speed': 20,
                    'suggested_questions': related_questions
                }
            })
        else:
            # Se não gerou perguntas, oferece ajuda genérica
            response_parts.append({
                'text': "\n\nPosso te ajudar com mais alguma informação sobre este assunto?",
                'metadata': {
                    'type_speed': 20
                }
            })
        
        # Feedback
        response_parts.append({
            'text': "\nEsta informação foi útil?",
            'metadata': {
                'buttons': [
                    {'title': '👍 Sim', 'payload': '/feedback_positive'},
                    {'title': '👎 Não', 'payload': '/feedback_negative'}
                ],
                'complete_before_next': True
            }
        })
        
        return response_parts

    def _format_faq_response(self, faq_match: Dict) -> List[Dict]:
        """Formata resposta para FAQ com perguntas sugeridas geradas dinamicamente"""
        response_parts = [
            {
                'text': f"❓ **Pergunta encontrada em {faq_match['filename']}:**\n{faq_match['question']}",
                'metadata': {
                    'response_part': 'question',
                    'complete_before_next': True
                }
            },
            {
                'text': f"✅ **Resposta completa:**\n{faq_match['answer']}",
                'metadata': {
                    'response_part': 'answer',
                    'complete_before_next': True
                }
            }
        ]
        
        # Gera perguntas relacionadas de forma inteligente
        related_questions = self._generate_related_questions(faq_match)
        
        if related_questions:
            questions_text = "\n\n🔍 Talvez você queira saber também sobre:"
            response_parts.append({
                'text': questions_text,
                'metadata': {
                    'type_speed': 20,
                    'suggested_questions': related_questions
                }
            })
        
        # Parte final com botões de feedback
        response_parts.append({
            'text': "\nEsta informação resolveu sua dúvida?",
            'metadata': {
                'response_part': 'confirmation',
                'buttons': [
                    {
                        'title': '👍 Sim',
                        'payload': '/feedback_positive'
                    },
                    {
                        'title': '👎 Não',
                        'payload': '/feedback_negative'
                    }
                ],
                'complete_before_next': True
            }
        })
        
        return response_parts

    def _format_faq_response(self, faq_match: Dict) -> List[Dict]:
        """Formata resposta para FAQ com botões de feedback e perguntas sugeridas"""
        response_parts = [
            {
                'text': f"❓ **Pergunta encontrada em {faq_match['filename']}:**\n{faq_match['question']}",
                'metadata': {
                    'response_part': 'question',
                    'complete_before_next': True
                }
            },
            {
                'text': f"✅ **Resposta completa:**\n{faq_match['answer']}",
                'metadata': {
                    'response_part': 'answer',
                    'complete_before_next': True
                }
            }
        ]
        
        # Adiciona informações extras para FAQs financeiras
        if faq_match.get('is_financial', False):
            response_parts.append({
                'text': "\n💡 Você também pode solicitar condições especiais diretamente com o serviço financeiro.",
                'metadata': {"type_speed": 20}
            })
        
        # Gera perguntas relacionadas
        related_questions = self._generate_related_questions(faq_match)
        if related_questions:
            questions_text = "\n\n🔍 Talvez você queira saber também:\n" + "\n".join(f"• {q}" for q in related_questions)
            response_parts.append({
                'text': questions_text,
                'metadata': {
                    'type_speed': 20,
                    'suggested_questions': related_questions
                }
            })
        
        # Parte final com botões de feedback
        response_parts.append({
            'text': "\nEsta informação resolveu sua dúvida?",
            'metadata': {
                'response_part': 'confirmation',
                'buttons': [
                    {
                        'title': '👍 Sim',
                        'payload': '/feedback_positive'
                    },
                    {
                        'title': '👎 Não',
                        'payload': '/feedback_negative'
                    }
                ],
                'complete_before_next': True
            }
        })
        
        return response_parts
    def _format_general_response(self, result: Dict, query: str) -> List[Dict]:
        """Formata resposta para conteúdo geral"""
        relevant = self._find_relevant_section(result['content'], query)
        
        # Resposta especializada para questões financeiras
        if self._is_financial_query(query):
            return self._format_financial_response(result)
        
        confidence = ""
        if result.get('similarity', 0) > 0.7:
            confidence = " (alta confiança)"
        elif result.get('similarity', 0) > 0.5:
            confidence = " (média confiança)"
        
        return [
            {
                'text': f"📄 **Informação encontrada em '{result['filename']}'{confidence}:**",
                'metadata': {"type_speed": 20}
            },
            {
                'text': f"\n\n{relevant['text']}",
                'metadata': {"type_speed": 15}
            },
            {
                'text': "\n\nPosso te ajudar com algo mais específico sobre este conteúdo?",
                'metadata': {"type_speed": 30, "delay": 1500}
            }
        ]


    def run(self, dispatcher, tracker, domain):
        query = tracker.latest_message.get('text', '').strip()
        self.logger.info(f"Processando consulta: '{query}'")
        
        try:
            # Busca híbrida - ambos os métodos em paralelo
            rag_results = self._semantic_search_rag(query, top_k=self.top_k)
            text_results = list(self.collection.find(
                {"$text": {"$search": query}},
                {"score": {"$meta": "textScore"}, "text_content": 1, "filename": 1}
            ).sort([("score", {"$meta": "textScore"})]).limit(3))
            
            # Combina e ordena resultados
            combined_results = []
            for result in rag_results:
                combined_results.append({
                    **result,
                    'type': 'semantic',
                    'combined_score': result.get('similarity', 0) * 0.7
                })
            
            for doc in text_results:
                combined_results.append({
                    'doc_id': doc['_id'],
                    'filename': doc.get('filename', ''),
                    'content': doc.get('text_content', ''),
                    'type': 'text',
                    'text_score': doc.get('score', 0),
                    'combined_score': doc.get('score', 0) * 0.3
                })
            
            # Remove duplicados mantendo a maior pontuação
            unique_results = {}
            for result in combined_results:
                doc_id = str(result.get('doc_id', ''))
                if doc_id not in unique_results or result['combined_score'] > unique_results[doc_id]['combined_score']:
                    unique_results[doc_id] = result
            
            sorted_results = sorted(unique_results.values(), key=lambda x: x['combined_score'], reverse=True)
            
             # Tratamento especial para FAQs
            if self._is_faq_query(query) and sorted_results:
                faq_match = self._find_best_faq_match(query, sorted_results)
                if faq_match:
                    response_parts = self._format_faq_response(faq_match)
                    for part in response_parts:
                        if part.get('metadata', {}).get('buttons'):
                            # Se tiver botões, envia como quick replies
                            dispatcher.utter_message(
                                text=part['text'],
                                buttons=part['metadata']['buttons']
                            )
                        else:
                            # Se não tiver botões, envia normalmente
                            dispatcher.utter_message(
                                text=part['text'],
                                metadata=part.get('metadata', {})
                            )
                    return []
            
            # Mostra o melhor resultado
            if sorted_results:
                best_result = sorted_results[0]
                
                # Se o melhor resultado tem baixa similaridade, pede confirmação
                if best_result.get('combined_score', 0) < 0.5:
                    dispatcher.utter_message(
                        text="Encontrei algumas informações que podem ser relevantes, mas não tenho certeza absoluta. Gostaria que eu mostrasse mesmo assim?",
                        metadata={
                            "buttons": [
                                {"title": "Sim", "payload": "/confirmar_sim"},
                                {"title": "Não", "payload": "/confirmar_nao"}
                            ]
                        }
                    )
                    return []
                
                for part in self._format_general_response(best_result, query):
                    dispatcher.utter_message(
                        text=part['text'],
                        metadata=part.get('metadata', {})
                    )
            else:
                dispatcher.utter_message(
                    text="Não encontrei informações sobre esse tópico. Poderia reformular sua pergunta com mais detalhes?",
                    metadata={"type_speed": 30}
                )
                
        except Exception as e:
            self.logger.error(f"Erro na busca: {str(e)}", exc_info=True)
            dispatcher.utter_message(
                text="Ocorreu um erro ao processar sua solicitação. Por favor, tente novamente mais tarde.",
                metadata={"type_speed": 30}
            )

        return []
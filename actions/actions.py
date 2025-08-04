from rasa_sdk import Action
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

logger = logging.getLogger(__name__)

class ActionSmartSearch(Action):
    def name(self):
        return "action_search_documents"

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.min_similarity = 0.4
        self.faq_threshold = 0.82
        self.top_k = 3
        self.context_window = 3
        
        # Conexão com MongoDB
        self.client = pymongo.MongoClient(
            "mongodb://root:root@uabbot-mongodb-1:27017/",
            serverSelectionTimeoutMS=5000
        )
        self.db = self.client["uab"]
        self.collection = self.db["documents"]
        
        # Modelos de embeddings
        self.sentence_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.roberta_model = RobertaModel.from_pretrained('roberta-base')
        
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
            
            # Gerar embeddings se necessário
            self._generate_document_embeddings()
            
        except Exception as e:
            self.logger.error(f"Erro na inicialização do RAG: {str(e)}")
            raise

    def _get_roberta_embedding(self, text: str) -> np.ndarray:
        """Gera embeddings com RoBERTa para o RAG"""
        inputs = self.roberta_tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512,
            padding='max_length'
        )
        with torch.no_grad():
            outputs = self.roberta_model(**inputs)
        return outputs.last_hidden_state[:, 0, :].numpy().squeeze()

    def _generate_document_embeddings(self):
        """Gera e armazena embeddings para documentos usando RoBERTa"""
        if self.db.document_embeddings.count_documents({}) > 0:
            return
            
        self.logger.info("Gerando embeddings RAG para documentos...")
        
        batch_size = 50
        total_docs = self.collection.count_documents({})
        processed = 0
        
        for i in range(0, total_docs, batch_size):
            batch = []
            documents = self.collection.find().skip(i).limit(batch_size)
            
            for doc in documents:
                content = doc.get("text_content", "")
                if not content:
                    continue
                    
                # Usar RoBERTa para embeddings RAG
                embedding = self._get_roberta_embedding(content[:1024])
                
                metadata = {
                    "doc_id": doc["_id"],
                    "filename": doc.get("filename", ""),
                    "embedding": embedding.tolist(),
                    "last_updated": datetime.now(),
                    "content_preview": content[:200] + "..." if len(content) > 200 else content
                }
                batch.append(metadata)
                processed += 1
            
            if batch:
                self.db.document_embeddings.insert_many(batch)
                self.logger.info(f"Progresso RAG: {processed}/{total_docs}")
        
        self.logger.info(f"Embeddings RAG gerados para {processed} documentos")

    def _semantic_search_rag(self, query: str, top_k: int = 3) -> List[Dict]:
        """Busca semântica usando RAG e RoBERTa"""
        query_embedding = self._get_roberta_embedding(query)
        
        # Pipeline de agregação para cálculo de similaridade no MongoDB
        pipeline = [
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
                                "query_norm": {"$sqrt": {"$reduce": {
                                    "input": query_embedding.tolist(),
                                    "initialValue": 0,
                                    "in": {"$add": ["$$value", {"$pow": ["$$this", 2]}]}
                                }}},
                                "embedding_norm": {"$sqrt": {"$reduce": {
                                    "input": "$embedding",
                                    "initialValue": 0,
                                    "in": {"$add": ["$$value", {"$pow": ["$$this", 2]}]}
                                }}}
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

    def _preprocess_text(self, text: str) -> str:
        """Pré-processa o texto para melhorar a correspondência"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)  # Remove pontuação
        return text.strip()

    def _extract_faqs(self, content: str) -> List[Dict]:
        """Extrai FAQs formatadas do conteúdo com múltiplos padrões melhorados"""
        faqs = []
        
        # Padrão 1: FAQs numeradas (1. Pergunta? Resposta...)
        pattern1 = r'(?:\d+[\.\)]\s+)?([^\n?]+\??)\s*([^\n]+(?:\n(?!\d+[\.\)]\s)[^\n]*)*)'
        
        # Padrão 2: Linhas com ? seguida de resposta
        pattern2 = r'([^\n]+\??)\s*([^\n]+(?:\n(?!\s*[^\n]+\??)[^\n]*)*)'
        
        # Padrão 3: FAQ com Q: / R: ou P: / R:
        pattern3 = r'(?:Pergunta|Q|P)\s*[:\.]\s*([^\n]+)\s*(?:Resposta|R|A)\s*[:\.]\s*([^\n]+(?:\n(?!(?:Pergunta|Q|P)\s*[:\.])[^\n]*)*)'
        
        # Padrão 4: Título seguido de resposta (para perguntas implícitas)
        pattern4 = r'(?:^|\n)\s*(?:-|\*)?\s*([^\n]+?)\s*[:\.]\s*([^\n]+(?:\n(?!\s*(?:-|\*)\s*[^\n]+[:\.])[^\n]*)*)'
        
        for pattern in [pattern1, pattern2, pattern3, pattern4]:
            matches = re.finditer(pattern, content, re.IGNORECASE | re.DOTALL)
            for match in matches:
                question = match.group(1).strip()
                answer = match.group(2).strip()
                
                # Verificar se parece uma pergunta-resposta válida
                if (len(question.split()) >= 3 and len(answer.split()) >= 5 and 
                    (any(q_word in question.lower() for q_word in ['como', 'qual', 'quando', 'onde', 'por que']) or 
                     '?' in question or 
                     len(question) < 100)):
                    
                    # Limpar a resposta
                    answer = re.sub(r'\s+', ' ', answer).strip()
                    
                    faqs.append({
                        'question': question,
                        'answer': answer,
                        'full_text': match.group(0).strip()
                    })
        
        return faqs

    def _is_faq_query(self, query: str) -> bool:
        """Determina se a consulta parece uma pergunta de FAQ com maior precisão"""
        question_words = ["como", "qual", "quando", "onde", "por que", "quais", "quanto", 
                        "contato", "telefone", "email", "endereço", "pode", "deve", "existe",
                        "posso", "preciso", "dificuldade", "problema", "ajuda", "dúvida"]
        
        query_lower = self._preprocess_text(query)
        
        # Verificar padrões de pergunta
        is_question = (
            any(query_lower.startswith(word) for word in question_words) or
            "?" in query or
            any(word in query_lower for word in [" o que ", " em que ", " para que "]) or
            re.search(r'\b(pode|deve|como)\s+[^\s]+\s+', query_lower) is not None
        )
        
        return is_question

    def _find_best_faq_match(self, query: str, docs: List[Dict]) -> Optional[Dict]:
        """Encontra a melhor correspondência de FAQ nos documentos usando abordagem aprimorada"""
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
                # Pré-processar perguntas e respostas
                clean_question = self._preprocess_text(faq['question'])
                clean_answer = self._preprocess_text(faq['answer'])
                clean_query = self._preprocess_text(query)
                
                # Verificar correspondência direta de palavras-chave
                keyword_match = (
                    any(word in clean_question for word in clean_query.split()[:5]) or
                    any(word in clean_answer for word in clean_query.split()[:5])
                )
                
                if not keyword_match:
                    continue
                
                # Calcula similaridade semântica
                question_emb = self.sentence_model.encode(clean_question)
                answer_emb = self.sentence_model.encode(clean_answer)
                
                question_sim = cosine_similarity([query_embedding], [question_emb])[0][0]
                answer_sim = cosine_similarity([query_embedding], [answer_emb])[0][0]
                similarity = max(question_sim, answer_sim)
                
                # Aumentar score se houver palavras-chave correspondentes
                if keyword_match:
                    similarity = min(similarity + 0.1, 1.0)
                
                if similarity > highest_score and similarity >= self.faq_threshold:
                    highest_score = similarity
                    best_match = {
                        'type': 'faq',
                        'question': faq['question'],
                        'answer': faq['answer'],
                        'score': similarity,
                        'filename': doc.get("filename", "documento"),
                        'content': content
                    }
        
        return best_match

    def _find_relevant_section(self, content: str, query: str) -> Dict:
        """Encontra a seção mais relevante no conteúdo para consultas não-FAQ"""
        paragraphs = [p.strip() for p in content.split('\n') if p.strip()]
        best_paragraph = ""
        best_score = 0
        
        query_embedding = self.sentence_model.encode(query)
        
        for para in paragraphs:
            if len(para.split()) < 10:  # Ignorar parágrafos muito curtos
                continue
                
            para_embedding = self.sentence_model.encode(para)
            similarity = cosine_similarity([query_embedding], [para_embedding])[0][0]
            
            # Bonus por correspondência de palavras-chave
            clean_para = self._preprocess_text(para)
            clean_query = self._preprocess_text(query)
            keyword_matches = sum(1 for word in clean_query.split() if word in clean_para)
            similarity = min(similarity + (keyword_matches * 0.05), 1.0)
            
            if similarity > best_score:
                best_score = similarity
                best_paragraph = para
        
        return {
            'text': best_paragraph if best_paragraph else content[:500],
            'score': best_score
        }

    def _format_faq_response(self, faq_match: Dict) -> str:
        """Formata resposta para FAQ encontrada, retornando sempre a resposta completa"""
        return (
            f"❓ **Pergunta encontrada em {faq_match['filename']}:**\n"
            f"{faq_match['question']}\n\n"
            f"✅ **Resposta completa:**\n"
            f"{faq_match['answer']}\n\n"
            f"Esta informação resolveu sua dúvida?"
        )

    def _format_general_response(self, result: Dict, query: str) -> str:
        """Formata resposta para conteúdo geral com seção relevante completa"""
        relevant = self._find_relevant_section(result['content'], query)
        
        confidence = ""
        if result['similarity'] > 0.7:
            confidence = " (alta confiança)"
        elif result['similarity'] > 0.5:
            confidence = " (média confiança)"
            
        return (
            f"📄 **Informação encontrada em '{result['filename']}'{confidence}:**\n\n"
            f"{relevant['text']}\n\n"
            f"Posso te ajudar com algo mais específico sobre este conteúdo?"
        )

    def _find_relevant_section(self, content: str, query: str) -> Dict:
        """Encontra a seção mais relevante mantendo a estrutura completa"""
        # Primeiro tenta encontrar por parágrafos
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        best_paragraph = ""
        best_score = 0
        
        query_embedding = self.sentence_model.encode(query)
        
        for para in paragraphs:
            if len(para.split()) < 5:  # Ignorar parágrafos muito curtos
                continue
                
            para_embedding = self.sentence_model.encode(para)
            similarity = cosine_similarity([query_embedding], [para_embedding])[0][0]
            
            # Bonus por correspondência de palavras-chave
            clean_para = self._preprocess_text(para)
            clean_query = self._preprocess_text(query)
            keyword_matches = sum(1 for word in clean_query.split() if word in clean_para)
            similarity = min(similarity + (keyword_matches * 0.05), 1.0)
            
            if similarity > best_score:
                best_score = similarity
                best_paragraph = para
        
        # Se encontrou um parágrafo bom, retorna ele
        if best_score > 0.5:
            return {'text': best_paragraph, 'score': best_score}
        
        # Se não, tenta encontrar por frases dentro do conteúdo
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', content) if s.strip()]
        best_sentence = ""
        best_sentence_score = 0
        
        for sentence in sentences:
            if len(sentence.split()) < 5:  # Ignorar frases muito curtas
                continue
                
            sentence_embedding = self.sentence_model.encode(sentence)
            similarity = cosine_similarity([query_embedding], [sentence_embedding])[0][0]
            
            if similarity > best_sentence_score:
                best_sentence_score = similarity
                best_sentence = sentence
        
        # Se encontrou uma frase boa, pega o contexto ao redor
        if best_sentence_score > 0.5:
            try:
                idx = sentences.index(best_sentence)
                start = max(0, idx - self.context_window)
                end = min(len(sentences), idx + self.context_window + 1)
                context = ' '.join(sentences[start:end])
                return {'text': context, 'score': best_sentence_score}
            except ValueError:
                pass
        
        # Fallback: retorna o início do conteúdo
        return {'text': content, 'score': 0.4}

    def run(self, dispatcher, tracker, domain):
        query = tracker.latest_message.get('text', '').strip()
        self.logger.info(f"Processando consulta: '{query}'")
        
        try:
            # Primeiro tenta buscar com RAG
            rag_results = self._semantic_search_rag(query, top_k=self.top_k)
            
            # Se for uma pergunta de FAQ, tenta encontrar correspondência exata
            if self._is_faq_query(query) and rag_results:
                faq_match = self._find_best_faq_match(query, rag_results)
                if faq_match:
                    dispatcher.utter_message(text=self._format_faq_response(faq_match))
                    return []
            
            # Se encontrou resultados RAG, mostra o mais relevante
            if rag_results:
                dispatcher.utter_message(text=self._format_general_response(rag_results[0], query))
                return []
            
            # Fallback para busca textual
            docs = list(self.collection.find(
                {"$text": {"$search": query}},
                {"score": {"$meta": "textScore"}, "text_content": 1, "filename": 1}
            ).sort([("score", {"$meta": "textScore"})]).limit(1))
            
            if docs:
                dispatcher.utter_message(
                    text=self._format_general_response({
                        'filename': docs[0]['filename'],
                        'content': docs[0]['text_content'],
                        'similarity': 0.5
                    }, query)
                )
            else:
                dispatcher.utter_message(text="Não encontrei informações sobre esse tópico. Poderia reformular sua pergunta?")
                
        except Exception as e:
            self.logger.error(f"Erro na busca: {str(e)}", exc_info=True)
            dispatcher.utter_message(
                text="Ocorreu um erro ao processar sua solicitação. Por favor, tente novamente."
            )

        return []
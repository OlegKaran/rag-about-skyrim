from __future__ import annotations

import os
from typing import List, Sequence, Tuple

import torch
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_ollama import ChatOllama
from langchain_qdrant import QdrantVectorStore
from sentence_transformers import CrossEncoder, SentenceTransformer

load_dotenv()

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "oblivion_lore")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
CROSS_ENCODER_MODEL_NAME = os.getenv("CROSS_ENCODER_MODEL_NAME", "BAAI/bge-reranker-v2-m3")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

DEFAULT_RETRIEVE_K = int(os.getenv("RETRIEVE_K", "20"))
DEFAULT_RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "5"))

SYSTEM_PROMPT_TEMPLATE = (
    "Ты эксперт по вселенной The Elder Scrolls IV: Oblivion. "
    "Используй только приведённые фрагменты лора, чтобы ответить на вопрос. "
    "Если ответа нет в контексте, честно скажи, что не знаешь. Не выдумывай информацию.\n\n"
    "Контекст:\n{context}"
)

def _select_device() -> str:
    override = os.getenv("MODEL_DEVICE", "auto")
    if override and override != "auto":
        return override
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"

class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str, device: str):
        self.model = SentenceTransformer(model_name, device=device)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

def format_context(documents: Sequence[Document]) -> str:
    if not documents:
        return "Контекст не найден."
    return "\n\n".join(f"Фрагмент {idx}:\n{doc.page_content}" for idx, doc in enumerate(documents, start=1))

def get_message_text(message) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(p.get("text", str(p)) if isinstance(p, dict) else str(p) for p in content)
    return str(content)

#-----------------------РЕТРИВЕР------------------
class RerankingRetriever:
    """Извлечение документов из Qdrant и их реранжирование.
    Не подгружает Ollama и Qwen.
    """
    def __init__(self):
        if not QDRANT_URL or not QDRANT_API_KEY:
            raise RuntimeError("Не заданы QDRANT_URL / QDRANT_API_KEY.")

        self.device = _select_device()
        print(f"[Retriever] Устройство: {self.device}")

        print(f"[Retriever] Загрузка эмбеддингов {EMBEDDING_MODEL_NAME}...")
        self.embeddings = SentenceTransformerEmbeddings(EMBEDDING_MODEL_NAME, device=self.device)

        print(f"[Retriever] Загрузка реранкера {CROSS_ENCODER_MODEL_NAME}...")
        self.cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL_NAME, device=self.device)

        print(f"[Retriever] Подключение к Qdrant, коллекция '{COLLECTION_NAME}'...")
        self.vector_store = QdrantVectorStore.from_existing_collection(
            embedding=self.embeddings,
            collection_name=COLLECTION_NAME,
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
        )

    def retrieve(self, query: str, retrieve_k: int = DEFAULT_RETRIEVE_K, top_n: int = DEFAULT_RERANK_TOP_N) -> List[Tuple[Document, float]]:
        """Достаёт документы из Qdrant и реранкает их."""
        documents = self.vector_store.similarity_search(query, k=retrieve_k)
        if not documents:
            return []
        pairs = [(query, doc.page_content) for doc in documents]
        scores = self.cross_encoder.predict(pairs)
        scored = sorted(zip(documents, scores), key=lambda x: float(x[1]), reverse=True)
        return [(doc, float(score)) for doc, score in scored[:top_n]]
    

#---------- ПОЛНЫЙ ПАЙЛПАЙН (РЕТРИВЕР + ГЕНЕРАТОР) --------------------------
class RAGPipeline:
    """Оркестратор: принимает готовый ретривер и добавляет к нему генерацию через LLM."""
    def __init__(self, retriever: RerankingRetriever):
        self.retriever = retriever
        
        print(f"[RAGPipeline] Подключение к Ollama ({OLLAMA_MODEL} @ {OLLAMA_BASE_URL})...")
        self.llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.0)
        print("[RAGPipeline] Пайплайн полностью готов к работе.")

    def answer(
        self,
        question: str,
        retrieve_k: int = DEFAULT_RETRIEVE_K,
        top_n: int = DEFAULT_RERANK_TOP_N,
        temperature: float = 0.0,
    ) -> Tuple[str, List[Tuple[Document, float]]]:

        # Используем выделенный компонент ретривера
        scored_docs = self.retriever.retrieve(question, retrieve_k=retrieve_k, top_n=top_n)
        docs = [doc for doc, _ in scored_docs]

        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=format_context(docs))
        self.llm.temperature = temperature
        response = self.llm.invoke([("system", system_prompt), ("human", question)])
        return get_message_text(response), scored_docs
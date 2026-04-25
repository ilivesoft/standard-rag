# FastAPI 앱 진입점 - RAG 파이프라인 API 서버
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.settings import settings
from pipeline.parser import DocumentParser
from pipeline.cleaner import TextCleaner
from pipeline.chunker import TextChunker
from pipeline.embedder import DocumentEmbedder
from pipeline.vectorstore_factory import create_vectorstore
from pipeline.vectorstore_protocol import VectorStoreProtocol
from pipeline.retriever import HybridRetriever
from pipeline.reranker import DocumentReranker
from pipeline.generator import ResponseGenerator
from pipeline.evaluator import RAGEvaluator
from pipeline.graphs.ingest_graph import create_ingest_graph
from pipeline.graphs.query_graph import create_query_graph

from api.ingest import router as ingest_router, set_ingest_graph
from api.query import (
    router as query_router,
    set_query_graph,
    set_dependencies,
    set_streaming_dependencies,
)
from api.evaluate import router as evaluate_router, set_evaluator
from api.index import router as index_router, set_vectorstore as set_index_vectorstore
from api.conversations import router as conversations_router, set_conversation_store
from pipeline.conversation_store import ConversationStore

# FastAPI 앱 초기화
app = FastAPI(
    title="RAG Pipeline API",
    description="End-to-End RAG Pipeline System - 문서 인덱싱 및 질의응답",
    version="1.0.0",
)

# CORS 미들웨어 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 파이프라인 컴포넌트 초기화
_parser = DocumentParser()
_cleaner = TextCleaner()
_chunker = TextChunker(
    chunk_size=settings.CHUNK_SIZE,
    chunk_overlap=settings.CHUNK_OVERLAP,
)
_embedder = DocumentEmbedder(
    model_name=settings.EMBEDDING_MODEL,
    device=settings.EMBEDDING_DEVICE,
)

# 환경 설정에 따라 적절한 벡터 백엔드 선택 및 초기화 (Fail-Fast 포함)
_selected_backend = settings.resolve_vectorstore_backend()
print(f"Vector backend: {_selected_backend}")
# import logging as _logging
# _logging.basicConfig(level=_logging.INFO)
# _app_logger = _logging.getLogger(__name__)
# _app_logger.info("Vector backend: %s", _selected_backend)

_vectorstore: VectorStoreProtocol = create_vectorstore(settings)
_retriever = HybridRetriever(vectorstore=_vectorstore, embedder=_embedder)
_reranker = DocumentReranker(model_name=settings.RERANKER_MODEL)
_generator = ResponseGenerator(
    provider=settings.LLM_PROVIDER,
    timeout=settings.LLM_TIMEOUT,
)
_evaluator = RAGEvaluator()

# 임베딩 모델 사전 로딩 (첫 요청 타임아웃 방지)
_embedder._load_model()

# LangGraph 워크플로우 생성
_ingest_graph = create_ingest_graph(
    _parser, _cleaner, _chunker, _embedder, _vectorstore
)
_query_graph = create_query_graph(
    _retriever, _reranker, _generator, _vectorstore, settings=settings
)

# 대화 저장소 초기화
_conversation_store = ConversationStore(db_path=settings.CONVERSATIONS_DB)

# API 라우터에 의존성 주입
set_ingest_graph(_ingest_graph)
set_query_graph(_query_graph)
set_dependencies(_vectorstore, _embedder, _reranker)
set_streaming_dependencies(_retriever, _generator)
set_evaluator(_evaluator)
set_index_vectorstore(_vectorstore)
set_conversation_store(_conversation_store)

# 라우터 등록
app.include_router(ingest_router)
app.include_router(query_router)
app.include_router(evaluate_router)
app.include_router(index_router)
app.include_router(conversations_router)


@app.get("/")
async def root():
    """루트 엔드포인트 - API 정보 반환"""
    return {
        "name": "RAG Pipeline API",
        "version": "1.0.0",
        "docs": "/docs",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

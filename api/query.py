# 질의 응답 API - /query, /query/stream, /collections, /health 엔드포인트
from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from api.conversations import get_conversation_store
from config.settings import settings
from models.request import QueryRequest
from models.response import QueryResponse, HealthResponse
from pipeline.vectorstore_protocol import VectorStoreProtocol

# 의존성은 main.py에서 주입
_query_graph = None
_vectorstore: VectorStoreProtocol | None = None
_embedder = None
_reranker = None
_retriever = None
_generator = None


def get_query_graph():
    return _query_graph


def set_query_graph(graph):
    global _query_graph
    _query_graph = graph


def set_dependencies(vectorstore: VectorStoreProtocol, embedder, reranker):
    """헬스체크용 의존성 설정"""
    global _vectorstore, _embedder, _reranker
    _vectorstore = vectorstore
    _embedder = embedder
    _reranker = reranker


def set_streaming_dependencies(retriever, generator):
    """SSE 스트리밍용 의존성 설정"""
    global _retriever, _generator
    _retriever = retriever
    _generator = generator


def _init_conversation_context(conversation_id: str) -> tuple[str, list[dict]]:
    """대화 ID를 확정하고 이전 turn history를 로드합니다.

    - conversation_id가 없으면 store에서 신규 대화를 생성
    - store가 없으면 (테스트 환경 등) 빈 결과 반환

    Returns:
        (확정된 conversation_id, chat_history 리스트)
    """
    store = get_conversation_store()
    if store is None:
        return "", []
    if not conversation_id:
        conv = store.create_conversation(title="")
        conversation_id = conv["id"]
    history = store.get_cached_history(conversation_id, max_turns=settings.HISTORY_TURNS)
    return conversation_id, history


def _build_query_state(request: QueryRequest, chat_history: list[dict]) -> dict:
    """LangGraph 초기 상태 딕셔너리를 생성합니다."""
    return {
        "query": request.query,
        "retrieved_chunks": [],
        "reranked_chunks": [],
        "answer": "",
        "sources": [],
        "error": None,
        "is_empty_collection": False,
        "top_k": request.top_k,
        "top_n": request.top_n,
        "alpha": request.alpha,
        "chat_history": chat_history,
        "quality_score": 0.0,
        "retry_count": 0,
    }


router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """질의에 대한 응답을 생성합니다."""
    graph = get_query_graph()
    if graph is None:
        raise HTTPException(status_code=500, detail="질의 그래프가 초기화되지 않았습니다.")

    conversation_id, chat_history = _init_conversation_context(request.conversation_id or "")
    result = graph.invoke(_build_query_state(request, chat_history))

    if result.get("error"):
        raise HTTPException(status_code=500, detail=result["error"])

    answer = result["answer"]
    store = get_conversation_store()
    if store is not None and conversation_id:
        store.append_turn(conversation_id, request.query, answer)

    return QueryResponse(
        answer=answer,
        sources=result["sources"],
        retrieved_count=len(result.get("retrieved_chunks", [])),
        reranked_count=len(result.get("reranked_chunks", [])),
        conversation_id=conversation_id or None,
    )


@router.post("/query/stream")
async def query_stream(request: QueryRequest):
    """SSE 스트리밍으로 질의 응답을 생성합니다."""
    if _retriever is None or _reranker is None or _generator is None:
        raise HTTPException(status_code=500, detail="스트리밍 의존성이 초기화되지 않았습니다.")

    if _vectorstore is not None and _vectorstore.count() == 0:
        async def empty_stream():
            yield "data: 인덱싱된 문서가 없습니다. 먼저 문서를 업로드하여 인덱싱해 주세요.\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(empty_stream(), media_type="text/event-stream")

    chunks = _retriever.retrieve(request.query, top_k=request.top_k, alpha=request.alpha)
    reranked = _reranker.rerank(request.query, chunks, top_n=request.top_n)

    conversation_id, chat_history = _init_conversation_context(request.conversation_id or "")
    store = get_conversation_store()

    async def token_stream():
        buffer: list[str] = []
        sources = [c.get("metadata", {}) for c in reranked]
        async for token in _generator.generate_stream(request.query, reranked, history=chat_history):
            buffer.append(token)
            yield f"data: {token}\n\n"
        if store is not None and conversation_id:
            store.append_turn(conversation_id, request.query, "".join(buffer))
        meta = json.dumps({
            "conversation_id": conversation_id,
            "sources": sources,
            "retrieved_count": len(chunks),
            "reranked_count": len(reranked),
        }, ensure_ascii=False)
        yield f"data: [META]{meta}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(token_stream(), media_type="text/event-stream")


@router.get("/collections")
async def list_collections():
    """사용 가능한 컬렉션 목록을 반환합니다."""
    if _vectorstore is None:
        return {"collections": []}
    return {"collections": _vectorstore.list_collections()}


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """시스템 헬스체크를 수행합니다."""
    chroma_status = "ok"
    if _vectorstore is not None:
        try:
            _vectorstore.count()
        except Exception:
            chroma_status = "error"

    embedding_status = "ok" if (_embedder is not None and _embedder.is_loaded) else "not_loaded"
    reranker_status = "ok" if (_reranker is not None and _reranker.is_loaded) else "not_loaded"

    return HealthResponse(
        status="ok" if chroma_status == "ok" else "degraded",
        chromadb_status=chroma_status,
        embedding_model_status=embedding_status,
        llm_status="unknown",
        reranker_status=reranker_status,
    )

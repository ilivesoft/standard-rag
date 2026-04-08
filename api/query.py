# 질의 응답 API - /query, /query/stream, /collections, /health 엔드포인트
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from models.request import QueryRequest
from models.response import QueryResponse, HealthResponse

# 의존성은 main.py에서 주입
_query_graph = None
_vectorstore = None
_embedder = None
_reranker = None
_retriever = None
_generator = None


def get_query_graph():
    return _query_graph


def set_query_graph(graph):
    global _query_graph
    _query_graph = graph


def set_dependencies(vectorstore, embedder, reranker):
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


router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """질의에 대한 응답을 생성합니다.

    Args:
        request: QueryRequest (query, top_k, top_n, alpha)

    Returns:
        QueryResponse: 생성된 답변과 출처
    """
    graph = get_query_graph()
    if graph is None:
        raise HTTPException(status_code=500, detail="질의 그래프가 초기화되지 않았습니다.")

    initial_state: dict = {
        "query": request.query,
        "query_embedding": [],
        "retrieved_chunks": [],
        "reranked_chunks": [],
        "answer": "",
        "sources": [],
        "error": None,
        "is_empty_collection": False,
        "top_k": request.top_k,
        "top_n": request.top_n,
        "alpha": request.alpha,
    }

    result = graph.invoke(initial_state)

    if result.get("error"):
        raise HTTPException(status_code=500, detail=result["error"])

    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"],
        retrieved_count=len(result.get("retrieved_chunks", [])),
        reranked_count=len(result.get("reranked_chunks", [])),
    )


@router.post("/query/stream")
async def query_stream(request: QueryRequest):
    """SSE 스트리밍으로 질의 응답을 생성합니다.

    Args:
        request: QueryRequest (query, top_k, top_n, alpha)

    Returns:
        StreamingResponse: text/event-stream 형식의 SSE 응답
    """
    if _retriever is None or _reranker is None or _generator is None:
        raise HTTPException(status_code=500, detail="스트리밍 의존성이 초기화되지 않았습니다.")

    if _vectorstore is not None and _vectorstore.count() == 0:
        async def empty_stream():
            yield "data: 인덱싱된 문서가 없습니다. 먼저 문서를 업로드하여 인덱싱해 주세요.\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(empty_stream(), media_type="text/event-stream")

    chunks = _retriever.retrieve(
        request.query,
        top_k=request.top_k,
        alpha=request.alpha,
    )
    reranked = _reranker.rerank(request.query, chunks, top_n=request.top_n)

    async def token_stream():
        async for token in _generator.generate_stream(request.query, reranked):
            yield f"data: {token}\n\n"
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
    # ChromaDB 상태 확인
    chroma_status = "ok"
    if _vectorstore is not None:
        try:
            _vectorstore.count()
        except Exception:
            chroma_status = "error"

    # 임베딩 모델 상태 확인
    embedding_status = "ok" if (_embedder is not None and _embedder.is_loaded) else "not_loaded"

    # Reranker 상태 확인
    reranker_status = "ok" if (_reranker is not None and _reranker.is_loaded) else "not_loaded"

    # LLM 상태 (연결 시도는 타임아웃이 있으므로 기본 상태로 설정)
    llm_status = "unknown"

    overall_status = "ok" if chroma_status == "ok" else "degraded"

    return HealthResponse(
        status=overall_status,
        chromadb_status=chroma_status,
        embedding_model_status=embedding_status,
        llm_status=llm_status,
        reranker_status=reranker_status,
    )

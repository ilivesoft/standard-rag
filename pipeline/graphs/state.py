# LangGraph 상태 정의 - TypedDict 기반 상태 스키마
from typing_extensions import TypedDict


class IngestState(TypedDict):
    """문서 인덱싱 워크플로우 상태"""
    file_path: str
    raw_text: str
    cleaned_text: str
    chunks: list[dict]
    embeddings: list[list[float]]
    stored_count: int
    error: str | None


class QueryState(TypedDict):
    """질의 응답 워크플로우 상태"""
    query: str
    retrieved_chunks: list[dict]
    reranked_chunks: list[dict]
    answer: str
    sources: list[dict]
    error: str | None
    is_empty_collection: bool
    # 검색 파라미터
    top_k: int
    top_n: int
    alpha: float
    # 대화 맥락 (이전 turn의 user/assistant 메시지, 시간 오름차순)
    chat_history: list[dict]
    # 검색 품질 평가
    quality_score: float   # reranked_chunks의 최고 rerank_score (grade_node 산출)
    retry_count: int       # 재검색 시도 횟수 (무한루프 방지용)

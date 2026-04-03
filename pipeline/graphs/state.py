# LangGraph 상태 정의 - TypedDict 기반 상태 스키마
from typing import Optional
from typing_extensions import TypedDict


class IngestState(TypedDict):
    """문서 인덱싱 워크플로우 상태"""
    file_path: str
    raw_text: str
    cleaned_text: str
    chunks: list[dict]
    embeddings: list[list[float]]
    stored_count: int
    error: Optional[str]


class QueryState(TypedDict):
    """질의 응답 워크플로우 상태"""
    query: str
    query_embedding: list[float]
    retrieved_chunks: list[dict]
    reranked_chunks: list[dict]
    answer: str
    sources: list[dict]
    error: Optional[str]
    is_empty_collection: bool
    # 검색 파라미터
    top_k: int
    top_n: int
    alpha: float

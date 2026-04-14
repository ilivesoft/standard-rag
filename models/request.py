# Pydantic v2 요청 스키마 - API 입력 모델
from pydantic import BaseModel, Field

from config.settings import settings


class QueryRequest(BaseModel):
    """질의 요청 모델"""
    query: str = Field(..., description="검색 질의 문자열")
    top_k: int = Field(default_factory=lambda: settings.TOP_K_RETRIEVAL, ge=1, le=100, description="검색 결과 상위 K개")
    top_n: int = Field(default_factory=lambda: settings.TOP_N_RERANK, ge=1, le=20, description="재순위 결과 상위 N개")
    alpha: float = Field(default_factory=lambda: settings.HYBRID_ALPHA, ge=0.0, le=1.0, description="하이브리드 검색 알파 (0=BM25, 1=Vector)")
    conversation_id: str | None = Field(
        default=None,
        description="기존 대화에 이어서 질의할 때 사용. 없거나 빈 문자열이면 백엔드가 신규 대화를 생성",
    )

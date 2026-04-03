# Pydantic v2 요청 스키마 - API 입력 모델
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """질의 요청 모델"""
    query: str = Field(..., description="검색 질의 문자열")
    top_k: int = Field(default=10, ge=1, le=100, description="검색 결과 상위 K개")
    top_n: int = Field(default=3, ge=1, le=20, description="재순위 결과 상위 N개")
    alpha: float = Field(default=0.5, ge=0.0, le=1.0, description="하이브리드 검색 알파 (0=BM25, 1=Vector)")

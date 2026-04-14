# Pydantic v2 응답 스키마 - API 출력 모델
from typing_extensions import TypedDict, NotRequired

from pydantic import BaseModel


class SourceMetadata(TypedDict, total=False):
    """질의 결과 출처 메타데이터"""
    source: str
    chunk_index: NotRequired[int]
    page: NotRequired[int]


class IngestResponse(BaseModel):
    """단일 파일 인덱싱 응답 모델"""
    filename: str
    chunks_indexed: int
    skipped: bool = False
    skip_reason: str | None = None


class FolderIngestResponse(BaseModel):
    """폴더 인덱싱 응답 모델"""
    total: int
    success: int
    failed: int
    results: list[IngestResponse]


class QueryResponse(BaseModel):
    """질의 응답 모델"""
    answer: str
    sources: list[SourceMetadata]
    retrieved_count: int
    reranked_count: int
    conversation_id: str | None = None


class HealthResponse(BaseModel):
    """헬스체크 응답 모델"""
    status: str
    chromadb_status: str
    embedding_model_status: str
    llm_status: str
    reranker_status: str


class DocumentSummary(BaseModel):
    """인덱스 문서 요약 모델"""
    source: str
    chunk_count: int


class IndexListResponse(BaseModel):
    """인덱스 문서 목록 응답 모델"""
    total_documents: int
    total_chunks: int
    documents: list[DocumentSummary]


class ChunkDetail(BaseModel):
    """청크 상세 정보 모델"""
    chunk_id: str
    text: str
    metadata: dict
    chunk_index: int


class IndexDetailResponse(BaseModel):
    """인덱스 문서 상세 응답 모델"""
    source: str
    chunk_count: int
    chunks: list[ChunkDetail]


class IndexDeleteResponse(BaseModel):
    """인덱스 삭제 응답 모델"""
    source: str
    deleted_count: int
    success: bool

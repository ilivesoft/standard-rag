# 인덱스 관리 API - /index/documents 엔드포인트
from fastapi import APIRouter, HTTPException
from models.response import (
    ChunkDetail,
    DocumentSummary,
    IndexDeleteResponse,
    IndexDetailResponse,
    IndexListResponse,
)

# 의존성은 main.py에서 주입
_vectorstore = None


def set_vectorstore(vs) -> None:
    """VectorStore 인스턴스를 설정합니다."""
    global _vectorstore
    _vectorstore = vs


def _get_vectorstore():
    """VectorStore 인스턴스를 반환합니다."""
    if _vectorstore is None:
        raise RuntimeError("VectorStore가 초기화되지 않았습니다.")
    return _vectorstore


router = APIRouter(prefix="/index", tags=["index"])


@router.get("/documents", response_model=IndexListResponse)
async def list_documents():
    """인덱싱된 모든 문서 목록을 반환합니다.

    Returns:
        IndexListResponse: 전체 문서 수, 전체 청크 수, 문서 목록
    """
    vectorstore = _get_vectorstore()

    # 소스별로 그룹화된 문서 목록 조회
    all_docs = vectorstore.get_all_documents()

    documents = [
        DocumentSummary(source=doc["source"], chunk_count=doc["chunk_count"])
        for doc in all_docs
    ]
    total_chunks = sum(doc["chunk_count"] for doc in all_docs)

    return IndexListResponse(
        total_documents=len(documents),
        total_chunks=total_chunks,
        documents=documents,
    )


@router.get("/documents/{source}", response_model=IndexDetailResponse)
async def get_document(source: str):
    """특정 문서의 청크 상세 정보를 반환합니다.

    Args:
        source: 조회할 파일명 (URL 경로 파라미터)

    Returns:
        IndexDetailResponse: 문서 소스, 청크 수, 청크 목록

    Raises:
        404: 해당 source의 청크를 찾을 수 없는 경우
    """
    vectorstore = _get_vectorstore()

    # source별 청크 조회
    chunks = vectorstore.get_by_source(source)

    if not chunks:
        raise HTTPException(
            status_code=404,
            detail=f"해당 문서를 찾을 수 없습니다: {source}",
        )

    chunk_details = [
        ChunkDetail(
            chunk_id=chunk.get("chunk_id", ""),
            text=chunk.get("text", ""),
            metadata=chunk.get("metadata", {}),
            chunk_index=chunk.get("metadata", {}).get("chunk_index", idx),
        )
        for idx, chunk in enumerate(chunks)
    ]

    return IndexDetailResponse(
        source=source,
        chunk_count=len(chunk_details),
        chunks=chunk_details,
    )


@router.delete("/documents/{source}", response_model=IndexDeleteResponse)
async def delete_document(source: str):
    """특정 문서를 인덱스에서 삭제합니다.

    Args:
        source: 삭제할 파일명 (URL 경로 파라미터)

    Returns:
        IndexDeleteResponse: 삭제 결과

    Raises:
        404: 해당 source의 문서를 찾을 수 없는 경우
    """
    vectorstore = _get_vectorstore()

    # 문서 삭제
    deleted_count = vectorstore.delete_by_source(source)

    if deleted_count == 0:
        raise HTTPException(
            status_code=404,
            detail=f"삭제할 문서를 찾을 수 없습니다: {source}",
        )

    return IndexDeleteResponse(
        source=source,
        deleted_count=deleted_count,
        success=True,
    )


@router.delete("/all", response_model=IndexDeleteResponse)
async def delete_all_documents():
    """인덱싱된 모든 데이터를 삭제합니다.

    Returns:
        IndexDeleteResponse: 삭제 결과 (source는 "*"로 표시)
    """
    vectorstore = _get_vectorstore()

    # 전체 삭제
    deleted_count = vectorstore.delete_all()

    return IndexDeleteResponse(
        source="*",
        deleted_count=deleted_count,
        success=True,
    )

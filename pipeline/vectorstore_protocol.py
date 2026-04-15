# 벡터 저장소 프로토콜 모듈 - 모든 벡터 백엔드가 구현해야 하는 인터페이스 계약
# @MX:ANCHOR: [AUTO] 모든 벡터 백엔드(ChromaVectorStore, PGVectorStore)가 의존하는 핵심 계약
# @MX:REASON: fan_in >= 3 (ChromaVectorStore, PGVectorStore, 팩토리, 소비자 다수)
from typing import Protocol, runtime_checkable


@runtime_checkable
class VectorStoreProtocol(Protocol):
    """모든 벡터 저장소 백엔드가 구현해야 하는 프로토콜 인터페이스.

    이 프로토콜은 ChromaDB와 PGVector 등 다양한 벡터 저장소 백엔드를
    동일한 인터페이스로 교체 사용할 수 있도록 추상화합니다.

    반환 스키마 통일:
    - search()는 항상 {text: str, metadata: dict, score: float} 형식 반환
    - score는 1 - cosine_distance 로 정규화 (0~1 범위, 높을수록 유사)
    """

    def add(self, chunks: list[dict], embeddings: list[list[float]]) -> int:
        """청크와 임베딩을 벡터 저장소에 저장합니다.

        Args:
            chunks: 저장할 청크 리스트. 각 항목은 'text'와 'metadata' 키를 포함.
            embeddings: 각 청크에 대응하는 임베딩 벡터 리스트.

        Returns:
            저장된 청크 수.
        """
        ...

    def search(self, query_embedding: list[float], top_k: int = 10) -> list[dict]:
        """벡터 유사도 검색을 수행합니다.

        Args:
            query_embedding: 질의 임베딩 벡터.
            top_k: 반환할 최대 결과 수.

        Returns:
            유사도 순으로 정렬된 딕셔너리 리스트.
            각 항목: {text: str, metadata: dict, score: float}
            score는 1 - cosine_distance (0~1 범위).
        """
        ...

    def count(self) -> int:
        """저장된 청크의 총 수를 반환합니다.

        Returns:
            현재 저장된 청크 수.
        """
        ...

    def get_all_texts(self) -> list[str]:
        """BM25 검색을 위해 저장된 모든 텍스트를 반환합니다.

        Returns:
            저장된 모든 청크 텍스트 리스트.
        """
        ...

    def get_all_texts_and_tokens(self) -> tuple[list[str], list[list[str]]]:
        """모든 텍스트와 사전 토큰화된 데이터를 반환합니다.

        Returns:
            (원본 텍스트 리스트, 토큰 리스트의 리스트) 튜플.
            토큰화 데이터가 없는 문서는 빈 리스트로 반환.
        """
        ...

    def list_collections(self) -> list[str]:
        """사용 가능한 컬렉션(테이블) 목록을 반환합니다.

        Returns:
            컬렉션 이름 문자열 리스트.
        """
        ...

    def get_all_documents(self) -> list[dict]:
        """저장된 모든 문서를 소스 파일 단위로 그룹화하여 반환합니다.

        Returns:
            소스 파일별 문서 리스트.
            각 항목: {source: str, chunk_count: int}
        """
        ...

    def get_by_source(self, source: str) -> list[dict]:
        """특정 소스 문서의 모든 청크를 반환합니다.

        Args:
            source: 조회할 소스 파일명.

        Returns:
            해당 소스의 청크 리스트.
            각 항목: {chunk_id: str, text: str, metadata: dict}
        """
        ...

    def delete_by_source(self, source: str) -> int:
        """특정 소스 문서의 모든 청크를 삭제합니다.

        Args:
            source: 삭제할 소스 파일명.

        Returns:
            삭제된 청크 수. 존재하지 않으면 0.
        """
        ...

    def delete_all(self) -> int:
        """저장된 모든 데이터를 삭제합니다.

        Returns:
            삭제된 청크 수.
        """
        ...

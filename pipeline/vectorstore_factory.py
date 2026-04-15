# 벡터 저장소 팩토리 모듈 - 환경 설정에 따라 적절한 백엔드를 생성합니다
# @MX:ANCHOR: [AUTO] main.py의 DI 유일 진입점 - 백엔드 선택 및 인스턴스 생성 담당
# @MX:REASON: main.py에서만 호출되는 단일 DI 진입점 (fan_in >= 3: main.py, 테스트 모듈)
from pipeline.vectorstore_protocol import VectorStoreProtocol
from pipeline.vectorstore_chromavector import ChromaVectorStore
from pipeline.vectorstore_pgvector import PGVectorStore


def create_vectorstore(settings) -> VectorStoreProtocol:
    """환경 설정에 따라 적절한 벡터 저장소 인스턴스를 생성하여 반환합니다.

    resolve_vectorstore_backend()를 통해 백엔드를 결정하고,
    해당 백엔드의 구현체를 생성합니다.

    Args:
        settings: pydantic-settings Settings 인스턴스.
                  APP_ENV, VECTORSTORE_BACKEND, CHROMA_*, PGVECTOR_* 필드를 참조.

    Returns:
        VectorStoreProtocol을 구현한 인스턴스.
        - chroma: ChromaVectorStore
        - pgvector: PGVectorStore

    Raises:
        ValueError: 지원하지 않는 VECTORSTORE_BACKEND 값인 경우.
        ConnectionError: pgvector 선택 시 PostgreSQL 연결 실패인 경우.

    Note:
        이 함수는 main.py와 테스트 모듈에서만 호출해야 합니다.
        API 라우터나 파이프라인 모듈이 직접 호출하면 DI 계층이 파괴됩니다.
    """
    # settings.resolve_vectorstore_backend()에서 ValueError 발생 가능
    backend = settings.resolve_vectorstore_backend()

    if backend == "chroma":
        return ChromaVectorStore(
            persist_dir=settings.CHROMA_PERSIST_DIR,
            collection=settings.CHROMA_COLLECTION,
        )

    if backend == "pgvector":
        return PGVectorStore(
            host=settings.PGVECTOR_HOST,
            port=settings.PGVECTOR_PORT,
            database=settings.PGVECTOR_DATABASE,
            user=settings.PGVECTOR_USER,
            password=settings.PGVECTOR_PASSWORD,
            collection=settings.PGVECTOR_COLLECTION,
            dimension=settings.PGVECTOR_DIMENSION,
        )

    # resolve_vectorstore_backend()가 검증하므로 여기에 도달하면 논리적 오류
    raise ValueError(f"Unsupported VECTORSTORE_BACKEND: {backend}")

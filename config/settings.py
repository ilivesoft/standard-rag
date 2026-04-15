# 전역 설정 모듈 - pydantic-settings 기반 환경 변수 관리
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """RAG 파이프라인 전역 설정 클래스.

    환경 변수로 모든 설정을 주입할 수 있습니다.
    APP_ENV로 벡터 백엔드를 결정합니다 (development→chroma, production→pgvector).
    """

    # LLM 설정
    LLM_PROVIDER: str = "ollama"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2"
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o-mini"
    LLM_TIMEOUT: int = 30

    # 임베딩 설정
    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    EMBEDDING_DEVICE: str = "cpu"

    # Reranker 설정
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # 환경 설정 - 벡터 백엔드 선택에 사용
    APP_ENV: str = "development"

    # ChromaDB 설정
    CHROMA_PERSIST_DIR: str = "./storage/chroma"
    CHROMA_COLLECTION: str = "default"

    # PGVector 설정 (운영 환경)
    PGVECTOR_HOST: str = "localhost"
    PGVECTOR_PORT: int = 5432
    PGVECTOR_DATABASE: str = "rag_vectors"
    PGVECTOR_USER: str = "rag_user"
    PGVECTOR_PASSWORD: str = ""  # 보안: 환경 변수로만 주입, 코드에 커밋 금지
    PGVECTOR_COLLECTION: str = "default"
    PGVECTOR_DIMENSION: int = 1024  # bge-m3 임베딩 차원

    # 청킹 설정
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64

    # 검색 설정
    TOP_K_RETRIEVAL: int = 10
    TOP_N_RERANK: int = 3
    HYBRID_ALPHA: float = 0.5  # 0=BM25, 1=벡터 검색

    # 배치 처리 설정
    BATCH_SIZE: int = 500
    MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100MB

    # OCR 설정
    OCR_LANGUAGES: list[str] = ["ko", "en"]
    OCR_GPU: bool = False

    # 파일 저장 경로
    UPLOAD_DIR: str = "./storage/uploads"

    # 대화 저장 경로
    CONVERSATIONS_DB: str = "./storage/conversations.db"

    # 채팅 응답 모드 (True=SSE 스트리밍, False=텍스트 일괄 응답)
    CHAT_STREAM: bool = False

    # 대화 맥락 설정 (LLM 프롬프트에 포함할 최근 turn 수, 1 turn = user+assistant 2 메시지)
    HISTORY_TURNS: int = 5

    # 평가 결과 저장 경로
    EVAL_RESULTS_DIR: str = "./storage/eval_results"

    def resolve_vectorstore_backend(self) -> str:
        """APP_ENV에 따라 벡터 백엔드를 결정하여 반환합니다.

        - development → chroma
        - production → pgvector

        Returns:
            선택된 백엔드 이름 ('chroma' 또는 'pgvector')

        Raises:
            ValueError: 알 수 없는 APP_ENV 값인 경우
        """
        env = self.APP_ENV.lower()
        if env == "development":
            return "chroma"
        if env == "production":
            return "pgvector"

        raise ValueError(
            f"알 수 없는 APP_ENV 값입니다: '{self.APP_ENV}'. "
            f"'development' 또는 'production'을 사용하세요."
        )


# 전역 싱글턴 인스턴스
settings = Settings()

# 전역 설정 모듈 - pydantic-settings 기반 환경 변수 관리
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """RAG 파이프라인 전역 설정 클래스"""

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

    # ChromaDB 설정
    CHROMA_PERSIST_DIR: str = "./storage/chroma"
    CHROMA_COLLECTION: str = "default"

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

    # 대화 맥락 설정 (LLM 프롬프트에 포함할 최근 turn 수, 1 turn = user+assistant 2 메시지)
    HISTORY_TURNS: int = 5

    # 평가 결과 저장 경로
    EVAL_RESULTS_DIR: str = "./storage/eval_results"



# 전역 싱글턴 인스턴스
settings = Settings()

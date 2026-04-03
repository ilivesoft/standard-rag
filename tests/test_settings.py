# 설정 모듈 테스트 - pydantic-settings 기반 환경 변수 관리 검증
from config.settings import Settings, settings


class TestSettingsDefaults:
    """Settings 기본값 검증"""

    def test_llm_provider_default(self):
        """LLM 제공자 기본값은 ollama"""
        s = Settings()
        assert s.LLM_PROVIDER == "ollama"

    def test_ollama_base_url_default(self):
        """Ollama 기본 URL 검증"""
        s = Settings()
        assert s.OLLAMA_BASE_URL == "http://localhost:11434"

    def test_ollama_model_default(self):
        """Ollama 모델 기본값 검증"""
        s = Settings()
        assert s.OLLAMA_MODEL == "llama3.2"

    def test_openai_api_key_default(self):
        """OpenAI API 키 기본값은 빈 문자열"""
        s = Settings()
        assert s.OPENAI_API_KEY == ""

    def test_openai_model_default(self):
        """OpenAI 모델 기본값 검증"""
        s = Settings()
        assert s.OPENAI_MODEL == "gpt-4o-mini"

    def test_llm_timeout_default(self):
        """LLM 타임아웃 기본값은 30초"""
        s = Settings()
        assert s.LLM_TIMEOUT == 30

    def test_embedding_model_default(self):
        """임베딩 모델 기본값 검증"""
        s = Settings()
        assert s.EMBEDDING_MODEL == "BAAI/bge-m3"

    def test_embedding_device_default(self):
        """임베딩 장치 기본값은 cpu"""
        s = Settings()
        assert s.EMBEDDING_DEVICE == "cpu"

    def test_reranker_model_default(self):
        """Reranker 모델 기본값 검증"""
        s = Settings()
        assert s.RERANKER_MODEL == "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def test_chroma_persist_dir_default(self):
        """ChromaDB 저장 경로 기본값 검증"""
        s = Settings()
        assert s.CHROMA_PERSIST_DIR == "./storage/chroma"

    def test_chroma_collection_default(self):
        """ChromaDB 컬렉션 기본값 검증"""
        s = Settings()
        assert s.CHROMA_COLLECTION == "default"

    def test_chunk_size_default(self):
        """청크 크기 기본값은 512"""
        s = Settings()
        assert s.CHUNK_SIZE == 512

    def test_chunk_overlap_default(self):
        """청크 오버랩 기본값은 64"""
        s = Settings()
        assert s.CHUNK_OVERLAP == 64

    def test_top_k_retrieval_default(self):
        """검색 상위 K 기본값은 10"""
        s = Settings()
        assert s.TOP_K_RETRIEVAL == 10

    def test_top_n_rerank_default(self):
        """Rerank 상위 N 기본값은 3"""
        s = Settings()
        assert s.TOP_N_RERANK == 3

    def test_hybrid_alpha_default(self):
        """하이브리드 검색 알파 기본값은 0.5"""
        s = Settings()
        assert s.HYBRID_ALPHA == 0.5

    def test_batch_size_default(self):
        """배치 크기 기본값은 500"""
        s = Settings()
        assert s.BATCH_SIZE == 500

    def test_max_file_size_default(self):
        """최대 파일 크기 기본값은 100MB"""
        s = Settings()
        assert s.MAX_FILE_SIZE == 100 * 1024 * 1024

    def test_upload_dir_default(self):
        """업로드 디렉토리 기본값 검증"""
        s = Settings()
        assert s.UPLOAD_DIR == "./storage/uploads"

    def test_eval_results_dir_default(self):
        """평가 결과 디렉토리 기본값 검증"""
        s = Settings()
        assert s.EVAL_RESULTS_DIR == "./storage/eval_results"


class TestSettingsEnvOverride:
    """환경 변수로 설정 오버라이드 검증"""

    def test_env_override_llm_provider(self, monkeypatch):
        """환경 변수로 LLM_PROVIDER 오버라이드 가능"""
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        s = Settings()
        assert s.LLM_PROVIDER == "openai"

    def test_env_override_chunk_size(self, monkeypatch):
        """환경 변수로 CHUNK_SIZE 오버라이드 가능"""
        monkeypatch.setenv("CHUNK_SIZE", "256")
        s = Settings()
        assert s.CHUNK_SIZE == 256

    def test_env_override_hybrid_alpha(self, monkeypatch):
        """환경 변수로 HYBRID_ALPHA 오버라이드 가능"""
        monkeypatch.setenv("HYBRID_ALPHA", "0.7")
        s = Settings()
        assert s.HYBRID_ALPHA == 0.7


class TestSettingsSingleton:
    """전역 settings 인스턴스 검증"""

    def test_global_settings_exists(self):
        """전역 settings 인스턴스가 존재"""
        assert settings is not None

    def test_global_settings_is_settings_instance(self):
        """전역 settings가 Settings 인스턴스"""
        assert isinstance(settings, Settings)

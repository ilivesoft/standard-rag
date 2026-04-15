# Settings 환경 설정 테스트 - 벡터 백엔드 자동 선택 및 검증 로직 검증
import pytest

from config.settings import Settings


class TestSettingsNewFields:
    """새로 추가된 환경 설정 필드들을 검증합니다."""

    def test_app_env_default_is_development(self):
        """APP_ENV 기본값은 'development'여야 합니다."""
        s = Settings()
        assert s.APP_ENV == "development"

    def test_vectorstore_backend_default_is_empty(self):
        """VECTORSTORE_BACKEND 기본값은 빈 문자열이어야 합니다."""
        s = Settings()
        assert s.VECTORSTORE_BACKEND == ""

    def test_pgvector_host_default(self):
        """PGVECTOR_HOST 기본값은 'localhost'여야 합니다."""
        s = Settings()
        assert s.PGVECTOR_HOST == "localhost"

    def test_pgvector_port_default(self):
        """PGVECTOR_PORT 기본값은 5432여야 합니다."""
        s = Settings()
        assert s.PGVECTOR_PORT == 5432

    def test_pgvector_database_default(self):
        """PGVECTOR_DATABASE 기본값은 'rag_vectors'여야 합니다."""
        s = Settings()
        assert s.PGVECTOR_DATABASE == "rag_vectors"

    def test_pgvector_user_default(self):
        """PGVECTOR_USER 기본값은 'rag_user'여야 합니다."""
        s = Settings()
        assert s.PGVECTOR_USER == "rag_user"

    def test_pgvector_password_default(self):
        """PGVECTOR_PASSWORD 기본값은 빈 문자열이어야 합니다."""
        s = Settings()
        assert s.PGVECTOR_PASSWORD == ""

    def test_pgvector_collection_default(self):
        """PGVECTOR_COLLECTION 기본값은 'default'여야 합니다."""
        s = Settings()
        assert s.PGVECTOR_COLLECTION == "default"

    def test_pgvector_dimension_default(self):
        """PGVECTOR_DIMENSION 기본값은 1024여야 합니다."""
        s = Settings()
        assert s.PGVECTOR_DIMENSION == 1024


class TestResolveVectorstoreBackend:
    """resolve_vectorstore_backend() 메서드의 자동 선택 및 검증 로직을 검증합니다."""

    @pytest.mark.parametrize("app_env,expected_backend", [
        ("development", "chroma"),
        ("Development", "chroma"),  # 대소문자 허용 (Edge 2 시나리오)
        ("DEVELOPMENT", "chroma"),
        ("production", "pgvector"),
        ("Production", "pgvector"),  # 대소문자 허용 (Edge 2 시나리오)
        ("PRODUCTION", "pgvector"),
    ])
    def test_auto_selection_by_app_env(self, app_env: str, expected_backend: str):
        """VECTORSTORE_BACKEND가 비어 있으면 APP_ENV에 따라 자동 선택됩니다."""
        s = Settings(APP_ENV=app_env, VECTORSTORE_BACKEND="")
        assert s.resolve_vectorstore_backend() == expected_backend

    def test_explicit_chroma_override(self):
        """VECTORSTORE_BACKEND=chroma이면 APP_ENV에 관계없이 chroma가 선택됩니다."""
        s = Settings(APP_ENV="production", VECTORSTORE_BACKEND="chroma")
        assert s.resolve_vectorstore_backend() == "chroma"

    def test_explicit_pgvector_override(self):
        """VECTORSTORE_BACKEND=pgvector이면 APP_ENV에 관계없이 pgvector가 선택됩니다."""
        s = Settings(APP_ENV="development", VECTORSTORE_BACKEND="pgvector")
        assert s.resolve_vectorstore_backend() == "pgvector"

    def test_explicit_backend_case_insensitive(self):
        """VECTORSTORE_BACKEND 값은 대소문자 구분 없이 처리됩니다."""
        s = Settings(VECTORSTORE_BACKEND="Chroma")
        assert s.resolve_vectorstore_backend() == "chroma"

    def test_unknown_backend_raises_value_error(self):
        """지원하지 않는 VECTORSTORE_BACKEND 값은 ValueError를 발생시킵니다."""
        s = Settings(VECTORSTORE_BACKEND="qdrant")
        with pytest.raises(ValueError, match="Unsupported VECTORSTORE_BACKEND: qdrant"):
            s.resolve_vectorstore_backend()

    def test_unknown_app_env_raises_value_error(self):
        """VECTORSTORE_BACKEND가 비어있고 APP_ENV가 알 수 없는 값이면 ValueError."""
        s = Settings(APP_ENV="staging", VECTORSTORE_BACKEND="")
        with pytest.raises(ValueError):
            s.resolve_vectorstore_backend()

    def test_development_env_selects_chroma_sanity(self):
        """개발 환경에서 Chroma가 선택되는 기본 동작 sanity 테스트."""
        s = Settings(APP_ENV="development")
        assert s.resolve_vectorstore_backend() == "chroma"

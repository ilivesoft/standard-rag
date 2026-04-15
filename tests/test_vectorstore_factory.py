# VectorStore 팩토리 테스트 - 환경별 벡터 백엔드 자동 선택 검증
import pytest
from unittest.mock import patch, MagicMock

from config.settings import Settings
from pipeline.vectorstore_factory import create_vectorstore
from pipeline.vectorstore_protocol import VectorStoreProtocol


class TestCreateVectorstoreChroma:
    """개발 환경에서 ChromaVectorStore가 생성되는지 검증합니다."""

    def test_development_env_creates_chroma(self, tmp_path):
        """APP_ENV=development이면 ChromaVectorStore를 반환해야 합니다."""
        from pipeline.vectorstore_chromavector import ChromaVectorStore
        s = Settings(
            APP_ENV="development",
            VECTORSTORE_BACKEND="",
            CHROMA_PERSIST_DIR=str(tmp_path / "chroma"),
            CHROMA_COLLECTION="test",
        )
        store = create_vectorstore(s)
        assert isinstance(store, ChromaVectorStore)

    def test_explicit_chroma_backend_creates_chroma(self, tmp_path):
        """VECTORSTORE_BACKEND=chroma이면 ChromaVectorStore를 반환해야 합니다."""
        from pipeline.vectorstore_chromavector import ChromaVectorStore
        s = Settings(
            APP_ENV="production",  # prod 환경이지만 명시적으로 chroma 선택
            VECTORSTORE_BACKEND="chroma",
            CHROMA_PERSIST_DIR=str(tmp_path / "chroma"),
            CHROMA_COLLECTION="test",
        )
        store = create_vectorstore(s)
        assert isinstance(store, ChromaVectorStore)

    def test_returned_store_is_protocol_instance(self, tmp_path):
        """create_vectorstore()가 반환하는 객체는 VectorStoreProtocol 인스턴스여야 합니다."""
        s = Settings(
            APP_ENV="development",
            CHROMA_PERSIST_DIR=str(tmp_path / "chroma"),
            CHROMA_COLLECTION="test",
        )
        store = create_vectorstore(s)
        assert isinstance(store, VectorStoreProtocol)


class TestCreateVectorstorePGVector:
    """운영 환경에서 PGVectorStore가 생성되는지 검증합니다 (PG 연결은 mock)."""

    def test_production_env_creates_pgvector(self):
        """APP_ENV=production이면 PGVectorStore 생성을 시도해야 합니다."""
        from pipeline.vectorstore_pgvector import PGVectorStore

        mock_store = MagicMock(spec=PGVectorStore)

        with patch("pipeline.vectorstore_factory.PGVectorStore", return_value=mock_store) as mock_class:
            s = Settings(
                APP_ENV="production",
                VECTORSTORE_BACKEND="",
                PGVECTOR_PASSWORD="testpwd",
            )
            create_vectorstore(s)

            # PGVectorStore가 올바른 인자로 호출되었는지 확인
            mock_class.assert_called_once_with(
                host=s.PGVECTOR_HOST,
                port=s.PGVECTOR_PORT,
                database=s.PGVECTOR_DATABASE,
                user=s.PGVECTOR_USER,
                password=s.PGVECTOR_PASSWORD,
                collection=s.PGVECTOR_COLLECTION,
                dimension=s.PGVECTOR_DIMENSION,
            )

    def test_explicit_pgvector_backend_creates_pgvector(self):
        """VECTORSTORE_BACKEND=pgvector이면 PGVectorStore 생성을 시도해야 합니다."""
        from pipeline.vectorstore_pgvector import PGVectorStore

        mock_store = MagicMock(spec=PGVectorStore)

        with patch("pipeline.vectorstore_factory.PGVectorStore", return_value=mock_store) as mock_class:
            s = Settings(
                APP_ENV="development",  # dev 환경이지만 명시적으로 pgvector 선택
                VECTORSTORE_BACKEND="pgvector",
                PGVECTOR_PASSWORD="testpwd",
            )
            create_vectorstore(s)
            assert mock_class.called


class TestCreateVectorstoreInvalidBackend:
    """지원하지 않는 백엔드 값에 대한 오류 처리 검증 (Scenario 4)."""

    def test_invalid_backend_raises_value_error(self):
        """지원하지 않는 VECTORSTORE_BACKEND 값은 ValueError를 발생시켜야 합니다."""
        s = Settings(VECTORSTORE_BACKEND="qdrant")
        with pytest.raises(ValueError, match="Unsupported VECTORSTORE_BACKEND: qdrant"):
            create_vectorstore(s)

    def test_error_message_contains_backend_name(self):
        """오류 메시지에 지원하지 않는 백엔드 이름이 포함되어야 합니다."""
        s = Settings(VECTORSTORE_BACKEND="weaviate")
        with pytest.raises(ValueError) as exc_info:
            create_vectorstore(s)
        assert "weaviate" in str(exc_info.value)

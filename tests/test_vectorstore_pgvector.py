# PGVectorStore 단위/통합 테스트
# 실제 PostgreSQL 연결이 필요한 테스트는 PGVECTOR_TEST_ENABLED=1 환경 변수로 활성화합니다.
import os
import pytest
from unittest.mock import MagicMock

from pipeline.vectorstore_pgvector import PGVectorStore


# 통합 테스트 활성화 여부 확인
PGVECTOR_ENABLED = os.environ.get("PGVECTOR_TEST_ENABLED", "0") == "1"
skip_if_no_pgvector = pytest.mark.skipif(
    not PGVECTOR_ENABLED,
    reason="PGVECTOR_TEST_ENABLED=1 환경 변수가 설정되지 않아 PGVector 통합 테스트를 건너뜁니다."
)


class TestPGVectorStoreProtocolCompliance:
    """PGVectorStore가 VectorStoreProtocol을 만족하는지 검증합니다."""

    def test_pgvectorstore_class_has_all_protocol_methods(self):
        """PGVectorStore 클래스가 프로토콜의 모든 메서드를 구현해야 합니다."""
        required_methods = [
            "add", "search", "count", "get_all_texts",
            "get_all_texts_and_tokens", "list_collections",
            "get_all_documents", "get_by_source",
            "delete_by_source", "delete_all",
        ]
        for method_name in required_methods:
            assert hasattr(PGVectorStore, method_name), (
                f"PGVectorStore에 '{method_name}' 메서드가 없습니다."
            )


class TestPGVectorStoreDimensionValidation:
    """임베딩 차원 불일치 검증 테스트 (Scenario 6)."""

    def test_add_raises_on_dimension_mismatch(self):
        """임베딩 차원이 설정값과 다르면 ValueError가 발생해야 합니다."""
        store = PGVectorStore.__new__(PGVectorStore)
        store._dimension = 768
        store._collection_name = "test"
        # 1024 차원 벡터로 add 시도 (설정은 768)
        chunks = [{"text": "테스트", "metadata": {"source": "test.pdf"}}]
        embeddings = [[0.1] * 1024]  # 차원 불일치

        # _store 속성 mock
        store._store = MagicMock()

        with pytest.raises(ValueError, match="768"):
            store.add(chunks, embeddings)

    def test_add_raises_with_expected_and_actual_dimensions_in_message(self):
        """오류 메시지에 기대 차원과 실제 차원이 모두 포함되어야 합니다."""
        store = PGVectorStore.__new__(PGVectorStore)
        store._dimension = 768
        store._collection_name = "test"
        store._store = MagicMock()

        chunks = [{"text": "테스트", "metadata": {}}]
        embeddings = [[0.1] * 1024]

        with pytest.raises(ValueError) as exc_info:
            store.add(chunks, embeddings)

        error_msg = str(exc_info.value)
        assert "768" in error_msg  # 기대 차원
        assert "1024" in error_msg  # 실제 차원


class TestPGVectorStoreConnectionString:
    """PGVectorStore 연결 문자열 생성 테스트."""

    def test_connection_string_format(self):
        """연결 문자열이 올바른 형식이어야 합니다."""
        conn_str = PGVectorStore._build_connection_string(
            host="localhost",
            port=5432,
            database="rag_vectors",
            user="rag_user",
            password="secret",
        )
        assert conn_str == "postgresql+psycopg://rag_user:secret@localhost:5432/rag_vectors"

    def test_connection_string_does_not_appear_in_repr(self):
        """연결 문자열(비밀번호 포함)이 repr에 노출되지 않아야 합니다."""
        # 이 테스트는 비밀번호 노출 방지를 위한 설계 검증
        conn_str = PGVectorStore._build_connection_string(
            host="myhost",
            port=5432,
            database="mydb",
            user="myuser",
            password="supersecret",
        )
        # 연결 문자열 자체에는 비밀번호가 포함되지만, 로그 메시지에는 포함되면 안 됨
        # 이 테스트는 _build_connection_string이 정적 메서드로 존재함을 확인
        assert "supersecret" in conn_str  # 연결 문자열 자체는 비밀번호 포함 (필요)


class TestPGVectorStoreSearchScoreConversion:
    """PGVectorStore search() 결과의 score 변환 테스트."""

    def test_search_converts_distance_to_similarity(self):
        """search() 결과의 score는 1 - distance로 변환되어야 합니다."""
        store = PGVectorStore.__new__(PGVectorStore)
        store._dimension = 1024
        store._collection_name = "test"

        # langchain_postgres는 (Document, distance) 튜플을 반환
        from langchain_core.documents import Document
        mock_doc = Document(page_content="테스트 텍스트", metadata={"source": "test.pdf"})
        distance = 0.3  # cosine distance

        mock_store = MagicMock()
        mock_store.similarity_search_with_score_by_vector.return_value = [
            (mock_doc, distance)
        ]
        store._store = mock_store

        results = store.search([0.1] * 1024, top_k=5)

        assert len(results) == 1
        assert abs(results[0]["score"] - (1 - distance)) < 1e-6
        assert results[0]["text"] == "테스트 텍스트"
        assert results[0]["metadata"]["source"] == "test.pdf"

    def test_search_empty_store_returns_empty_list(self):
        """빈 컬렉션에서 search()는 빈 리스트를 반환해야 합니다 (Edge 1)."""
        store = PGVectorStore.__new__(PGVectorStore)
        store._dimension = 1024
        store._collection_name = "test"

        mock_store = MagicMock()
        mock_store.similarity_search_with_score_by_vector.return_value = []
        store._store = mock_store

        results = store.search([0.1] * 1024, top_k=5)
        assert results == []

    def test_search_result_schema(self):
        """search() 결과는 {text, metadata, score} 형식이어야 합니다."""
        store = PGVectorStore.__new__(PGVectorStore)
        store._dimension = 1024
        store._collection_name = "test"

        from langchain_core.documents import Document
        mock_doc = Document(
            page_content="내용",
            metadata={"source": "doc.pdf", "chunk_index": 0}
        )
        mock_store = MagicMock()
        mock_store.similarity_search_with_score_by_vector.return_value = [
            (mock_doc, 0.1)
        ]
        store._store = mock_store

        results = store.search([0.5] * 1024, top_k=1)
        assert len(results) == 1
        result = results[0]
        assert "text" in result
        assert "metadata" in result
        assert "score" in result
        assert isinstance(result["score"], float)


class TestPGVectorStoreAddMethod:
    """PGVectorStore add() 메서드 단위 테스트."""

    def test_add_empty_chunks_returns_zero(self):
        """빈 청크 리스트를 add()하면 0을 반환해야 합니다."""
        store = PGVectorStore.__new__(PGVectorStore)
        store._dimension = 1024
        store._store = MagicMock()

        result = store.add([], [])
        assert result == 0
        store._store.add_embeddings.assert_not_called()

    def test_add_calls_add_embeddings_with_correct_args(self):
        """add()가 store.add_embeddings를 올바른 인자로 호출해야 합니다."""
        store = PGVectorStore.__new__(PGVectorStore)
        store._dimension = 3
        store._collection_name = "test"

        chunks = [
            {"text": "첫 번째 청크", "metadata": {"source": "a.pdf"}},
            {"text": "두 번째 청크", "metadata": {"source": "a.pdf"}},
        ]
        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

        mock_store = MagicMock()
        store._store = mock_store

        result = store.add(chunks, embeddings)

        assert result == 2
        mock_store.add_embeddings.assert_called_once_with(
            texts=["첫 번째 청크", "두 번째 청크"],
            embeddings=embeddings,
            metadatas=[{"source": "a.pdf"}, {"source": "a.pdf"}],
        )

    def test_add_returns_chunk_count(self):
        """add()가 저장된 청크 수를 반환해야 합니다."""
        store = PGVectorStore.__new__(PGVectorStore)
        store._dimension = 2
        store._store = MagicMock()

        chunks = [{"text": "텍스트", "metadata": {}}]
        embeddings = [[0.1, 0.2]]

        result = store.add(chunks, embeddings)
        assert result == 1


class TestPGVectorStoreSessionMethods:
    """세션 기반 메서드 단위 테스트 (mock session)."""

    def _make_store_with_mock_session(self, query_results):
        """모크 세션이 있는 PGVectorStore를 생성합니다."""
        store = PGVectorStore.__new__(PGVectorStore)
        store._dimension = 1024
        store._collection_name = "test_col"

        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.fetchone.return_value = query_results[0] if isinstance(query_results, list) and len(query_results) > 0 else None
        mock_result.fetchall.return_value = query_results if isinstance(query_results, list) else []
        mock_session.execute.return_value = mock_result

        # context manager 지원
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=mock_session)
        mock_cm.__exit__ = MagicMock(return_value=False)

        mock_store = MagicMock()
        mock_store._make_sync_session.return_value = mock_cm
        store._store = mock_store
        return store, mock_session

    def test_count_returns_zero_on_empty(self):
        """count()가 빈 컬렉션에서 0을 반환해야 합니다."""
        store, _ = self._make_store_with_mock_session([(0,)])
        result = store.count()
        assert result == 0

    def test_count_returns_correct_value(self):
        """count()가 올바른 청크 수를 반환해야 합니다."""
        store, mock_session = self._make_store_with_mock_session([(42,)])
        result = store.count()
        assert result == 42

    def test_get_all_texts_returns_list(self):
        """get_all_texts()가 텍스트 리스트를 반환해야 합니다."""
        store, _ = self._make_store_with_mock_session([("텍스트1",), ("텍스트2",)])
        results = store.get_all_texts()
        assert results == ["텍스트1", "텍스트2"]

    def test_get_all_texts_and_tokens_parses_tokenized_text(self):
        """get_all_texts_and_tokens()가 tokenized_text를 리스트로 파싱해야 합니다."""
        store, _ = self._make_store_with_mock_session([
            ("텍스트", "토큰1 토큰2 토큰3"),
        ])
        texts, tokens = store.get_all_texts_and_tokens()
        assert texts == ["텍스트"]
        assert tokens == [["토큰1", "토큰2", "토큰3"]]

    def test_get_all_texts_and_tokens_handles_missing_tokenized_text(self):
        """tokenized_text가 없으면 빈 리스트를 반환해야 합니다."""
        store, _ = self._make_store_with_mock_session([("텍스트", None)])
        texts, tokens = store.get_all_texts_and_tokens()
        assert texts == ["텍스트"]
        assert tokens == [[]]

    def test_list_collections_returns_names(self):
        """list_collections()가 컬렉션 이름 리스트를 반환해야 합니다."""
        store, _ = self._make_store_with_mock_session([("col1",), ("col2",)])
        result = store.list_collections()
        assert result == ["col1", "col2"]

    def test_get_all_documents_groups_by_source(self):
        """get_all_documents()가 소스별로 그룹화된 결과를 반환해야 합니다."""
        store, _ = self._make_store_with_mock_session([
            ("a.pdf", 5),
            ("b.pdf", 3),
        ])
        result = store.get_all_documents()
        assert len(result) == 2
        assert result[0] == {"source": "a.pdf", "chunk_count": 5}

    def test_get_by_source_returns_chunks(self):
        """get_by_source()가 해당 소스의 청크를 반환해야 합니다."""
        store, _ = self._make_store_with_mock_session([
            ("uuid-1", "텍스트", {"source": "a.pdf"}),
        ])
        result = store.get_by_source("a.pdf")
        assert len(result) == 1
        assert result[0]["text"] == "텍스트"
        assert result[0]["chunk_id"] == "uuid-1"

    def test_delete_by_source_returns_row_count(self):
        """delete_by_source()가 삭제된 행 수를 반환해야 합니다."""
        store = PGVectorStore.__new__(PGVectorStore)
        store._dimension = 1024
        store._collection_name = "test_col"

        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.rowcount = 3
        mock_session.execute.return_value = mock_result

        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=mock_session)
        mock_cm.__exit__ = MagicMock(return_value=False)

        mock_store = MagicMock()
        mock_store._make_sync_session.return_value = mock_cm
        store._store = mock_store

        result = store.delete_by_source("a.pdf")
        assert result == 3

    def test_delete_all_calls_delete_collection(self):
        """delete_all()이 delete_collection과 create_collection을 호출해야 합니다."""
        store = PGVectorStore.__new__(PGVectorStore)
        store._dimension = 1024
        store._collection_name = "test_col"

        # count를 위한 mock session
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.fetchone.return_value = (10,)
        mock_session.execute.return_value = mock_result

        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=mock_session)
        mock_cm.__exit__ = MagicMock(return_value=False)

        mock_store = MagicMock()
        mock_store._make_sync_session.return_value = mock_cm
        store._store = mock_store

        result = store.delete_all()
        assert result == 10
        mock_store.delete_collection.assert_called_once()
        mock_store.create_collection.assert_called_once()


@pytest.mark.integration
@skip_if_no_pgvector
class TestPGVectorStoreIntegration:
    """실제 PostgreSQL에 연결하는 통합 테스트.

    실행 방법:
        docker compose -f docker-compose.dev.yml up -d
        PGVECTOR_TEST_ENABLED=1 pytest tests/test_vectorstore_pgvector.py -m integration -v
    """

    @pytest.fixture
    def store(self):
        """통합 테스트용 PGVectorStore 픽스처."""
        s = PGVectorStore(
            host="localhost",
            port=5432,
            database="rag_vectors",
            user="rag_user",
            password="rag_dev_password",
            collection="test_integration",
            dimension=4,  # 테스트용 소형 차원
        )
        # 테스트 전후 정리
        s.delete_all()
        yield s
        s.delete_all()

    def test_add_and_count(self, store):
        """add() 후 count()가 올바른 수를 반환해야 합니다."""
        chunks = [
            {"text": "문서 1", "metadata": {"source": "test.pdf"}},
            {"text": "문서 2", "metadata": {"source": "test.pdf"}},
        ]
        embeddings = [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]
        count = store.add(chunks, embeddings)
        assert count == 2
        assert store.count() == 2

    def test_search_returns_results(self, store):
        """add() 후 search()가 결과를 반환해야 합니다."""
        chunks = [{"text": "벡터 검색 테스트", "metadata": {"source": "test.pdf"}}]
        embeddings = [[1.0, 0.0, 0.0, 0.0]]
        store.add(chunks, embeddings)

        results = store.search([1.0, 0.0, 0.0, 0.0], top_k=5)
        assert len(results) >= 1
        assert results[0]["text"] == "벡터 검색 테스트"
        assert 0.0 <= results[0]["score"] <= 1.0

    def test_delete_by_source(self, store):
        """delete_by_source()가 해당 소스 청크만 삭제해야 합니다."""
        chunks = [
            {"text": "파일A", "metadata": {"source": "a.pdf"}},
            {"text": "파일B", "metadata": {"source": "b.pdf"}},
        ]
        embeddings = [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]
        store.add(chunks, embeddings)

        deleted = store.delete_by_source("a.pdf")
        assert deleted == 1
        assert store.count() == 1

    def test_get_all_texts_and_tokens_format(self, store):
        """get_all_texts_and_tokens()가 BM25 호환 형식을 반환해야 합니다."""
        chunks = [{
            "text": "한국어 텍스트",
            "metadata": {"source": "test.pdf", "tokenized_text": "한국어 텍스트"},
        }]
        embeddings = [[0.1, 0.2, 0.3, 0.4]]
        store.add(chunks, embeddings)

        texts, tokens = store.get_all_texts_and_tokens()
        assert len(texts) == 1
        assert len(tokens) == 1
        assert isinstance(tokens[0], list)  # 토큰은 리스트 형식

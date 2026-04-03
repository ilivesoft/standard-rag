# 벡터 저장소 모듈 테스트 - tmp_path 픽스처, CRUD 검증
import pytest
import numpy as np
from pipeline.vectorstore import VectorStore


@pytest.fixture
def vector_store(tmp_path):
    """tmp_path를 사용하는 격리된 VectorStore 인스턴스"""
    return VectorStore(
        persist_dir=str(tmp_path / "chroma"),
        collection="test_collection"
    )


@pytest.fixture
def sample_chunks():
    """테스트용 샘플 청크"""
    return [
        {"text": "파이썬은 인터프리터 언어입니다.", "metadata": {"source": "doc1.txt", "chunk_index": 0}},
        {"text": "머신러닝은 데이터로부터 학습합니다.", "metadata": {"source": "doc1.txt", "chunk_index": 1}},
        {"text": "딥러닝은 신경망을 사용합니다.", "metadata": {"source": "doc2.txt", "chunk_index": 0}},
    ]


@pytest.fixture
def sample_embeddings():
    """테스트용 고정 임베딩"""
    np.random.seed(42)
    return [np.random.rand(384).tolist() for _ in range(3)]


class TestVectorStoreAdd:
    """add 메서드 테스트"""

    def test_add_returns_count(self, vector_store, sample_chunks, sample_embeddings):
        """add가 저장된 청크 수를 반환"""
        result = vector_store.add(sample_chunks, sample_embeddings)
        assert result == 3

    def test_add_increases_count(self, vector_store, sample_chunks, sample_embeddings):
        """add 후 count가 증가"""
        assert vector_store.count() == 0
        vector_store.add(sample_chunks, sample_embeddings)
        assert vector_store.count() == 3

    def test_add_empty_list_returns_zero(self, vector_store):
        """빈 리스트 추가 시 0 반환"""
        result = vector_store.add([], [])
        assert result == 0


class TestVectorStoreSearch:
    """search 메서드 테스트"""

    def test_search_returns_list(self, vector_store, sample_chunks, sample_embeddings):
        """search가 리스트를 반환"""
        vector_store.add(sample_chunks, sample_embeddings)
        query_embedding = np.random.rand(384).tolist()
        result = vector_store.search(query_embedding, top_k=2)
        assert isinstance(result, list)

    def test_search_respects_top_k(self, vector_store, sample_chunks, sample_embeddings):
        """search가 top_k 개수를 준수"""
        vector_store.add(sample_chunks, sample_embeddings)
        query_embedding = np.random.rand(384).tolist()
        result = vector_store.search(query_embedding, top_k=2)
        assert len(result) <= 2

    def test_search_result_has_text(self, vector_store, sample_chunks, sample_embeddings):
        """검색 결과에 text 키가 존재"""
        vector_store.add(sample_chunks, sample_embeddings)
        query_embedding = np.random.rand(384).tolist()
        result = vector_store.search(query_embedding, top_k=1)
        for item in result:
            assert "text" in item

    def test_search_result_has_metadata(self, vector_store, sample_chunks, sample_embeddings):
        """검색 결과에 metadata 키가 존재"""
        vector_store.add(sample_chunks, sample_embeddings)
        query_embedding = np.random.rand(384).tolist()
        result = vector_store.search(query_embedding, top_k=1)
        for item in result:
            assert "metadata" in item

    def test_search_empty_store_returns_empty(self, vector_store):
        """빈 저장소 검색 시 빈 리스트 반환"""
        query_embedding = np.random.rand(384).tolist()
        result = vector_store.search(query_embedding, top_k=5)
        assert result == []


class TestVectorStoreGetAllTexts:
    """get_all_texts 메서드 테스트"""

    def test_get_all_texts_returns_list(self, vector_store, sample_chunks, sample_embeddings):
        """get_all_texts가 리스트를 반환"""
        vector_store.add(sample_chunks, sample_embeddings)
        result = vector_store.get_all_texts()
        assert isinstance(result, list)

    def test_get_all_texts_correct_count(self, vector_store, sample_chunks, sample_embeddings):
        """get_all_texts가 저장된 모든 텍스트 반환"""
        vector_store.add(sample_chunks, sample_embeddings)
        result = vector_store.get_all_texts()
        assert len(result) == 3


class TestVectorStoreCount:
    """count 메서드 테스트"""

    def test_count_zero_initially(self, vector_store):
        """초기 상태에서 count가 0"""
        assert vector_store.count() == 0

    def test_count_after_add(self, vector_store, sample_chunks, sample_embeddings):
        """add 후 count가 정확"""
        vector_store.add(sample_chunks, sample_embeddings)
        assert vector_store.count() == 3

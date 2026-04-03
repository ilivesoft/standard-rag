# 하이브리드 검색 모듈 테스트 - Vector, BM25, RRF 병합 검증
import pytest
import numpy as np
from unittest.mock import MagicMock
from pipeline.retriever import HybridRetriever


@pytest.fixture
def mock_vectorstore():
    """VectorStore 모킹"""
    vs = MagicMock()
    vs.search.return_value = [
        {"text": "파이썬 텍스트", "metadata": {"source": "doc1.txt"}, "score": 0.9},
        {"text": "머신러닝 텍스트", "metadata": {"source": "doc2.txt"}, "score": 0.8},
    ]
    vs.get_all_texts.return_value = [
        "파이썬은 인터프리터 언어입니다.",
        "머신러닝은 데이터 분석에 사용됩니다.",
    ]
    return vs


@pytest.fixture
def mock_embedder():
    """DocumentEmbedder 모킹"""
    emb = MagicMock()
    emb.embed_query.return_value = np.random.rand(384).tolist()
    return emb


@pytest.fixture
def retriever(mock_vectorstore, mock_embedder):
    """HybridRetriever 인스턴스"""
    return HybridRetriever(vectorstore=mock_vectorstore, embedder=mock_embedder)


class TestHybridRetrieverRetrieve:
    """retrieve 메서드 테스트"""

    def test_retrieve_returns_list(self, retriever):
        """retrieve가 리스트를 반환"""
        result = retriever.retrieve("파이썬 검색")
        assert isinstance(result, list)

    def test_retrieve_calls_embedder(self, retriever, mock_embedder):
        """retrieve가 embedder.embed_query를 호출"""
        retriever.retrieve("테스트 쿼리")
        mock_embedder.embed_query.assert_called_once()

    def test_retrieve_calls_vector_search(self, retriever, mock_vectorstore):
        """retrieve가 vectorstore.search를 호출"""
        retriever.retrieve("테스트 쿼리")
        mock_vectorstore.search.assert_called_once()

    def test_retrieve_result_has_text(self, retriever):
        """검색 결과 각 항목에 text 키가 존재"""
        result = retriever.retrieve("파이썬")
        for item in result:
            assert "text" in item


class TestHybridRetrieverRRF:
    """RRF (Reciprocal Rank Fusion) 테스트"""

    def test_rrf_fusion_returns_list(self, retriever):
        """_rrf_fusion이 리스트를 반환"""
        vector_results = [
            {"text": "텍스트 A", "metadata": {}},
            {"text": "텍스트 B", "metadata": {}},
        ]
        bm25_results = [
            {"text": "텍스트 B", "metadata": {}},
            {"text": "텍스트 C", "metadata": {}},
        ]
        result = retriever._rrf_fusion(vector_results, bm25_results)
        assert isinstance(result, list)

    def test_rrf_fusion_deduplicates(self, retriever):
        """_rrf_fusion이 중복 텍스트를 제거"""
        same_results = [
            {"text": "같은 텍스트", "metadata": {}},
        ]
        result = retriever._rrf_fusion(same_results, same_results)
        texts = [r["text"] for r in result]
        assert len(texts) == len(set(texts))

    def test_rrf_empty_inputs(self, retriever):
        """_rrf_fusion이 빈 입력을 처리"""
        result = retriever._rrf_fusion([], [])
        assert result == []


class TestHybridRetrieverBM25:
    """BM25 검색 테스트"""

    def test_bm25_search_returns_list(self, retriever):
        """_bm25_search가 리스트를 반환"""
        result = retriever._bm25_search("파이썬", top_k=2)
        assert isinstance(result, list)

    def test_bm25_empty_corpus(self, mock_embedder):
        """빈 코퍼스에서 BM25 검색 시 빈 리스트"""
        vs = MagicMock()
        vs.get_all_texts.return_value = []
        retriever = HybridRetriever(vectorstore=vs, embedder=mock_embedder)
        result = retriever._bm25_search("쿼리", top_k=5)
        assert result == []

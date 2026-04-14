# Reranker 모듈 테스트 - HuggingFaceCrossEncoder 모킹, 재순위 결과 검증
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from pipeline.reranker import DocumentReranker


@pytest.fixture
def mock_reranker():
    """HuggingFaceCrossEncoder를 모킹한 DocumentReranker"""
    with patch("pipeline.reranker.HuggingFaceCrossEncoder") as mock_cls:
        mock_instance = MagicMock()
        # score는 scores 리스트를 반환
        mock_instance.score.return_value = np.array([0.9, 0.3, 0.7])
        mock_cls.return_value = mock_instance

        reranker = DocumentReranker(model_name="test-reranker")
        # Lazy load 트리거
        reranker.rerank("쿼리", [
            {"text": "텍스트 1", "metadata": {}},
            {"text": "텍스트 2", "metadata": {}},
            {"text": "텍스트 3", "metadata": {}},
        ], top_n=2)
        yield reranker, mock_instance


@pytest.fixture
def sample_chunks():
    """재순위용 샘플 청크"""
    return [
        {"text": "파이썬 프로그래밍 언어", "metadata": {"source": "doc1.txt"}},
        {"text": "자바 프로그래밍 언어", "metadata": {"source": "doc2.txt"}},
        {"text": "머신러닝 알고리즘", "metadata": {"source": "doc3.txt"}},
    ]


class TestDocumentRerankerRerank:
    """rerank 메서드 테스트"""

    def test_rerank_returns_list(self, mock_reranker, sample_chunks):
        """rerank가 리스트를 반환"""
        reranker, mock_model = mock_reranker
        mock_model.score.return_value = np.array([0.5, 0.8, 0.3])
        result = reranker.rerank("프로그래밍", sample_chunks, top_n=2)
        assert isinstance(result, list)

    def test_rerank_respects_top_n(self, mock_reranker, sample_chunks):
        """rerank가 top_n 개수를 준수"""
        reranker, mock_model = mock_reranker
        mock_model.score.return_value = np.array([0.5, 0.8, 0.3])
        result = reranker.rerank("프로그래밍", sample_chunks, top_n=2)
        assert len(result) == 2

    def test_rerank_sorted_by_score_descending(self, mock_reranker, sample_chunks):
        """재순위 결과가 점수 내림차순으로 정렬"""
        reranker, mock_model = mock_reranker
        mock_model.score.return_value = np.array([0.5, 0.9, 0.3])
        result = reranker.rerank("프로그래밍", sample_chunks, top_n=3)
        scores = [r.get("rerank_score", 0) for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_rerank_highest_score_first(self, mock_reranker, sample_chunks):
        """가장 높은 점수의 청크가 첫 번째"""
        reranker, mock_model = mock_reranker
        mock_model.score.return_value = np.array([0.1, 0.95, 0.4])
        result = reranker.rerank("질문", sample_chunks, top_n=1)
        assert result[0]["text"] == "자바 프로그래밍 언어"

    def test_rerank_empty_chunks_returns_empty(self, mock_reranker):
        """빈 청크 리스트에서 rerank 시 빈 리스트"""
        reranker, mock_model = mock_reranker
        mock_model.score.return_value = np.array([])
        result = reranker.rerank("쿼리", [], top_n=3)
        assert result == []


class TestDocumentRerankerLazyLoad:
    """Lazy Loading 테스트"""

    def test_model_not_loaded_at_init(self):
        """초기화 시 모델이 즉시 로드되지 않음"""
        with patch("pipeline.reranker.HuggingFaceCrossEncoder") as mock_cls:
            DocumentReranker(model_name="test-model")
            mock_cls.assert_not_called()

    def test_is_loaded_false_before_use(self):
        """사용 전 is_loaded가 False"""
        with patch("pipeline.reranker.HuggingFaceCrossEncoder"):
            reranker = DocumentReranker(model_name="test-model")
            assert reranker.is_loaded is False

    def test_is_loaded_true_after_rerank(self, sample_chunks):
        """rerank 호출 후 is_loaded가 True"""
        with patch("pipeline.reranker.HuggingFaceCrossEncoder") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.score.return_value = np.array([0.5, 0.8, 0.3])
            mock_cls.return_value = mock_instance
            reranker = DocumentReranker(model_name="test-model")
            reranker.rerank("쿼리", sample_chunks, top_n=2)
            assert reranker.is_loaded is True

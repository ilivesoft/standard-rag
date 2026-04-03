# 임베딩 모듈 테스트 - SentenceTransformer 모킹, 벡터 차원 검증
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from pipeline.embedder import DocumentEmbedder


@pytest.fixture
def mock_embedder():
    """SentenceTransformer를 모킹한 DocumentEmbedder"""
    with patch("pipeline.embedder.SentenceTransformer") as mock_cls:
        mock_instance = MagicMock()
        # 고정 384차원 벡터 반환
        def fake_encode(texts, **kwargs):
            if isinstance(texts, str):
                return np.random.rand(384)
            return np.array([np.random.rand(384) for _ in texts])

        mock_instance.encode.side_effect = fake_encode
        mock_cls.return_value = mock_instance
        embedder = DocumentEmbedder(model_name="test-model", device="cpu")
        # 강제로 모델 로드 트리거
        _ = embedder.embed(["dummy"])
        yield embedder, mock_instance


class TestDocumentEmbedderEmbed:
    """embed 메서드 테스트"""

    def test_embed_returns_list(self, mock_embedder):
        """embed가 리스트를 반환"""
        embedder, _ = mock_embedder
        result = embedder.embed(["텍스트 1", "텍스트 2"])
        assert isinstance(result, list)

    def test_embed_correct_count(self, mock_embedder):
        """입력 개수와 동일한 임베딩 반환"""
        embedder, _ = mock_embedder
        texts = ["텍스트 1", "텍스트 2", "텍스트 3"]
        result = embedder.embed(texts)
        assert len(result) == 3

    def test_embed_each_is_list_of_floats(self, mock_embedder):
        """각 임베딩이 float 리스트"""
        embedder, _ = mock_embedder
        result = embedder.embed(["텍스트"])
        assert isinstance(result[0], list)
        assert all(isinstance(v, float) for v in result[0])

    def test_embed_correct_dimension(self, mock_embedder):
        """임베딩 차원이 384"""
        embedder, _ = mock_embedder
        result = embedder.embed(["텍스트"])
        assert len(result[0]) == 384

    def test_embed_empty_list_returns_empty(self, mock_embedder):
        """빈 리스트 입력 시 빈 리스트 반환"""
        embedder, mock_model = mock_embedder
        mock_model.encode.return_value = np.array([])
        result = embedder.embed([])
        assert result == []


class TestDocumentEmbedderEmbedQuery:
    """embed_query 메서드 테스트"""

    def test_embed_query_returns_list(self, mock_embedder):
        """embed_query가 리스트를 반환"""
        embedder, mock_model = mock_embedder
        mock_model.encode.return_value = np.random.rand(384)
        result = embedder.embed_query("검색 질문")
        assert isinstance(result, list)

    def test_embed_query_correct_dimension(self, mock_embedder):
        """embed_query 결과가 384차원"""
        embedder, mock_model = mock_embedder
        mock_model.encode.return_value = np.random.rand(384)
        result = embedder.embed_query("검색 질문")
        assert len(result) == 384


class TestDocumentEmbedderLazyLoad:
    """Lazy Loading 테스트"""

    def test_model_not_loaded_at_init(self):
        """초기화 시 모델이 즉시 로드되지 않음"""
        with patch("pipeline.embedder.SentenceTransformer") as mock_cls:
            DocumentEmbedder(model_name="test-model")
            mock_cls.assert_not_called()

    def test_is_loaded_false_before_use(self):
        """사용 전 is_loaded가 False"""
        with patch("pipeline.embedder.SentenceTransformer"):
            embedder = DocumentEmbedder(model_name="test-model")
            assert embedder.is_loaded is False

    def test_is_loaded_true_after_embed(self):
        """embed 호출 후 is_loaded가 True"""
        with patch("pipeline.embedder.SentenceTransformer") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.encode.return_value = np.array([np.random.rand(384)])
            mock_cls.return_value = mock_instance
            embedder = DocumentEmbedder(model_name="test-model")
            embedder.embed(["텍스트"])
            assert embedder.is_loaded is True

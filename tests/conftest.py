# pytest 공통 픽스처 모음
import pytest
import numpy as np
import chromadb
from unittest.mock import MagicMock, patch


@pytest.fixture
def tmp_chroma_client(tmp_path):
    """테스트용 임시 ChromaDB 에피머럴 클라이언트"""
    client = chromadb.EphemeralClient()
    yield client


@pytest.fixture
def sample_chunks():
    """테스트용 샘플 청크 리스트"""
    return [
        {"text": "파이썬은 인터프리터 언어입니다.", "metadata": {"source": "test.txt", "chunk_index": 0}},
        {"text": "머신러닝은 데이터로부터 학습합니다.", "metadata": {"source": "test.txt", "chunk_index": 1}},
        {"text": "딥러닝은 신경망을 사용합니다.", "metadata": {"source": "test.txt", "chunk_index": 2}},
    ]


@pytest.fixture
def sample_embeddings():
    """테스트용 고정 임베딩 벡터 (384차원)"""
    np.random.seed(42)
    return [np.random.rand(384).tolist() for _ in range(3)]


@pytest.fixture
def sample_texts():
    """테스트용 샘플 텍스트 리스트"""
    return [
        "파이썬은 인터프리터 언어입니다.",
        "머신러닝은 데이터로부터 학습합니다.",
        "딥러닝은 신경망을 사용합니다.",
    ]


@pytest.fixture
def mock_sentence_transformer():
    """SentenceTransformer 모킹 픽스처"""
    with patch("sentence_transformers.SentenceTransformer") as mock_cls:
        mock_instance = MagicMock()
        # encode 호출 시 고정 numpy 배열 반환
        mock_instance.encode.return_value = np.array([
            np.random.rand(384),
            np.random.rand(384),
            np.random.rand(384),
        ])
        mock_cls.return_value = mock_instance
        yield mock_cls, mock_instance


@pytest.fixture
def sample_txt_file(tmp_path):
    """테스트용 임시 TXT 파일"""
    f = tmp_path / "sample.txt"
    f.write_text("안녕하세요. 이것은 테스트 텍스트입니다.", encoding="utf-8")
    return f


@pytest.fixture
def sample_md_file(tmp_path):
    """테스트용 임시 MD 파일"""
    f = tmp_path / "sample.md"
    f.write_text("# 제목\n\n이것은 마크다운 파일입니다.", encoding="utf-8")
    return f


@pytest.fixture
def sample_html_file(tmp_path):
    """테스트용 임시 HTML 파일"""
    f = tmp_path / "sample.html"
    f.write_text("<html><body><h1>제목</h1><p>이것은 HTML 파일입니다.</p></body></html>", encoding="utf-8")
    return f

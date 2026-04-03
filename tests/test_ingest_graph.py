# LangGraph 인덱싱 그래프 테스트 - 노드 상태 전달, 실행 순서 검증
import pytest
import numpy as np
from unittest.mock import MagicMock
from pipeline.graphs.ingest_graph import create_ingest_graph


@pytest.fixture
def mock_parser(tmp_path):
    """DocumentParser 모킹"""
    parser = MagicMock()
    # parse는 텍스트 문자열을 반환
    parser.parse.return_value = "파싱된 텍스트 내용입니다."
    parser.is_supported.return_value = True
    return parser


@pytest.fixture
def mock_cleaner():
    """TextCleaner 모킹"""
    cleaner = MagicMock()
    cleaner.clean.return_value = "정제된 텍스트 내용입니다."
    return cleaner


@pytest.fixture
def mock_chunker():
    """TextChunker 모킹"""
    chunker = MagicMock()
    chunker.chunk.return_value = [
        {"text": "청크 1", "metadata": {"source": "test.txt", "chunk_index": 0}},
        {"text": "청크 2", "metadata": {"source": "test.txt", "chunk_index": 1}},
    ]
    return chunker


@pytest.fixture
def mock_embedder():
    """DocumentEmbedder 모킹"""
    embedder = MagicMock()
    embedder.embed.return_value = [
        np.random.rand(384).tolist(),
        np.random.rand(384).tolist(),
    ]
    return embedder


@pytest.fixture
def mock_vectorstore():
    """VectorStore 모킹"""
    vs = MagicMock()
    vs.add.return_value = 2
    return vs


@pytest.fixture
def ingest_graph(mock_parser, mock_cleaner, mock_chunker, mock_embedder, mock_vectorstore):
    """테스트용 IngestGraph"""
    return create_ingest_graph(mock_parser, mock_cleaner, mock_chunker, mock_embedder, mock_vectorstore)


class TestIngestGraphExecution:
    """IngestGraph 실행 테스트"""

    def test_graph_runs_successfully(self, ingest_graph, tmp_path):
        """그래프가 오류 없이 실행"""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("테스트")

        initial_state = {
            "file_path": str(txt_file),
            "raw_text": "",
            "cleaned_text": "",
            "chunks": [],
            "embeddings": [],
            "stored_count": 0,
            "error": None,
        }
        result = ingest_graph.invoke(initial_state)
        assert result is not None

    def test_graph_parse_node_called(self, ingest_graph, mock_parser, tmp_path):
        """parse 노드가 호출됨"""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("테스트")

        initial_state = {
            "file_path": str(txt_file),
            "raw_text": "",
            "cleaned_text": "",
            "chunks": [],
            "embeddings": [],
            "stored_count": 0,
            "error": None,
        }
        ingest_graph.invoke(initial_state)
        mock_parser.parse.assert_called_once()

    def test_graph_clean_node_called(self, ingest_graph, mock_cleaner, tmp_path):
        """clean 노드가 호출됨"""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("테스트")

        initial_state = {
            "file_path": str(txt_file),
            "raw_text": "",
            "cleaned_text": "",
            "chunks": [],
            "embeddings": [],
            "stored_count": 0,
            "error": None,
        }
        ingest_graph.invoke(initial_state)
        mock_cleaner.clean.assert_called_once()

    def test_graph_stored_count_in_result(self, ingest_graph, tmp_path):
        """결과에 stored_count가 포함"""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("테스트")

        initial_state = {
            "file_path": str(txt_file),
            "raw_text": "",
            "cleaned_text": "",
            "chunks": [],
            "embeddings": [],
            "stored_count": 0,
            "error": None,
        }
        result = ingest_graph.invoke(initial_state)
        assert "stored_count" in result

    def test_graph_stored_count_matches(self, ingest_graph, tmp_path):
        """stored_count가 vectorstore.add 반환값과 일치"""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("테스트")

        initial_state = {
            "file_path": str(txt_file),
            "raw_text": "",
            "cleaned_text": "",
            "chunks": [],
            "embeddings": [],
            "stored_count": 0,
            "error": None,
        }
        result = ingest_graph.invoke(initial_state)
        assert result["stored_count"] == 2

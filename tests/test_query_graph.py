# LangGraph 질의 그래프 테스트 - 정상 흐름, 빈 컬렉션 조건부 라우팅 검증
import pytest
from unittest.mock import MagicMock
from pipeline.graphs.query_graph import create_query_graph


@pytest.fixture
def mock_retriever():
    """HybridRetriever 모킹"""
    r = MagicMock()
    r.retrieve.return_value = [
        {"text": "검색된 청크 1", "metadata": {"source": "doc1.txt"}},
        {"text": "검색된 청크 2", "metadata": {"source": "doc2.txt"}},
    ]
    return r


@pytest.fixture
def mock_reranker():
    """DocumentReranker 모킹"""
    r = MagicMock()
    r.rerank.return_value = [
        {"text": "재순위된 청크 1", "metadata": {"source": "doc1.txt"}, "rerank_score": 0.9},
    ]
    return r


@pytest.fixture
def mock_generator():
    """ResponseGenerator 모킹"""
    g = MagicMock()
    g.generate.return_value = {
        "answer": "파이썬은 범용 프로그래밍 언어입니다.",
        "sources": [{"source": "doc1.txt"}],
    }
    return g


@pytest.fixture
def mock_vectorstore_with_docs():
    """문서가 있는 VectorStore 모킹"""
    vs = MagicMock()
    vs.count.return_value = 5
    return vs


@pytest.fixture
def mock_vectorstore_empty():
    """빈 VectorStore 모킹"""
    vs = MagicMock()
    vs.count.return_value = 0
    return vs


@pytest.fixture
def base_state():
    """기본 QueryState"""
    return {
        "query": "파이썬이란?",
        "query_embedding": [],
        "retrieved_chunks": [],
        "reranked_chunks": [],
        "answer": "",
        "sources": [],
        "error": None,
        "is_empty_collection": False,
        "top_k": 10,
        "top_n": 3,
        "alpha": 0.5,
    }


class TestQueryGraphNormalFlow:
    """정상 흐름 테스트"""

    def test_graph_runs_successfully(
        self, mock_retriever, mock_reranker, mock_generator, mock_vectorstore_with_docs, base_state
    ):
        """그래프가 오류 없이 실행"""
        graph = create_query_graph(mock_retriever, mock_reranker, mock_generator, mock_vectorstore_with_docs)
        result = graph.invoke(base_state)
        assert result is not None

    def test_graph_result_has_answer(
        self, mock_retriever, mock_reranker, mock_generator, mock_vectorstore_with_docs, base_state
    ):
        """결과에 answer가 포함"""
        graph = create_query_graph(mock_retriever, mock_reranker, mock_generator, mock_vectorstore_with_docs)
        result = graph.invoke(base_state)
        assert "answer" in result

    def test_graph_result_has_sources(
        self, mock_retriever, mock_reranker, mock_generator, mock_vectorstore_with_docs, base_state
    ):
        """결과에 sources가 포함"""
        graph = create_query_graph(mock_retriever, mock_reranker, mock_generator, mock_vectorstore_with_docs)
        result = graph.invoke(base_state)
        assert "sources" in result

    def test_graph_calls_retriever(
        self, mock_retriever, mock_reranker, mock_generator, mock_vectorstore_with_docs, base_state
    ):
        """retriever가 호출됨"""
        graph = create_query_graph(mock_retriever, mock_reranker, mock_generator, mock_vectorstore_with_docs)
        graph.invoke(base_state)
        mock_retriever.retrieve.assert_called_once()

    def test_graph_calls_reranker(
        self, mock_retriever, mock_reranker, mock_generator, mock_vectorstore_with_docs, base_state
    ):
        """reranker가 호출됨"""
        graph = create_query_graph(mock_retriever, mock_reranker, mock_generator, mock_vectorstore_with_docs)
        graph.invoke(base_state)
        mock_reranker.rerank.assert_called_once()

    def test_graph_calls_generator(
        self, mock_retriever, mock_reranker, mock_generator, mock_vectorstore_with_docs, base_state
    ):
        """generator가 호출됨"""
        graph = create_query_graph(mock_retriever, mock_reranker, mock_generator, mock_vectorstore_with_docs)
        graph.invoke(base_state)
        mock_generator.generate.assert_called_once()


class TestQueryGraphEmptyCollection:
    """빈 컬렉션 라우팅 테스트"""

    def test_empty_collection_skips_retriever(
        self, mock_retriever, mock_reranker, mock_generator, mock_vectorstore_empty, base_state
    ):
        """빈 컬렉션일 때 retriever가 호출되지 않음 (또는 no_docs 경로로 분기)"""
        graph = create_query_graph(mock_retriever, mock_reranker, mock_generator, mock_vectorstore_empty)
        result = graph.invoke(base_state)
        # 빈 컬렉션일 때 기본 안내 메시지 반환
        assert "answer" in result
        assert result["answer"] != ""

    def test_empty_collection_answer_message(
        self, mock_retriever, mock_reranker, mock_generator, mock_vectorstore_empty, base_state
    ):
        """빈 컬렉션일 때 안내 메시지가 answer에 포함"""
        graph = create_query_graph(mock_retriever, mock_reranker, mock_generator, mock_vectorstore_empty)
        result = graph.invoke(base_state)
        # 인덱싱된 문서가 없다는 메시지
        assert "인덱싱" in result["answer"] or "문서" in result["answer"]

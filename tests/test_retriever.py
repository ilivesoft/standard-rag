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
        "자바스크립트는 웹 개발에 필수입니다.",
        "데이터베이스 설계는 중요합니다.",
    ]
    # 인제스트 시 사전 토큰화된 데이터 (조사 제거 완료)
    vs.get_all_texts_and_tokens.return_value = (
        [
            "파이썬은 인터프리터 언어입니다.",
            "머신러닝은 데이터 분석에 사용됩니다.",
            "자바스크립트는 웹 개발에 필수입니다.",
            "데이터베이스 설계는 중요합니다.",
        ],
        [
            ["파이썬", "인터프리터", "언어"],
            ["머신러닝", "데이터", "분석", "사용"],
            ["자바스크립트", "웹", "개발", "필수"],
            ["데이터베이스", "설계", "중요"],
        ],
    )
    vs.count.return_value = 4
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

    def test_rrf_alpha_vector_only(self, retriever):
        """alpha=1.0이면 Vector 결과만 RRF 점수에 반영"""
        vector_results = [{"text": "벡터 결과", "metadata": {}}]
        bm25_results = [{"text": "BM25 결과", "metadata": {}}]
        result = retriever._rrf_fusion(vector_results, bm25_results, alpha=1.0)
        scores = {r["text"]: r["rrf_score"] for r in result}
        assert scores["벡터 결과"] > 0.0
        assert scores["BM25 결과"] == 0.0

    def test_rrf_alpha_bm25_only(self, retriever):
        """alpha=0.0이면 BM25 결과만 RRF 점수에 반영"""
        vector_results = [{"text": "벡터 결과", "metadata": {}}]
        bm25_results = [{"text": "BM25 결과", "metadata": {}}]
        result = retriever._rrf_fusion(vector_results, bm25_results, alpha=0.0)
        scores = {r["text"]: r["rrf_score"] for r in result}
        assert scores["벡터 결과"] == 0.0
        assert scores["BM25 결과"] > 0.0

    def test_rrf_alpha_balanced(self, retriever):
        """alpha=0.5이면 동일 순위 결과의 가중치가 같음"""
        vector_results = [{"text": "같은 텍스트", "metadata": {}}]
        bm25_results = [{"text": "같은 텍스트", "metadata": {}}]
        result = retriever._rrf_fusion(vector_results, bm25_results, alpha=0.5)
        # alpha=0.5: 양쪽 동일 가중치, 점수 = 0.5/(60+1) + 0.5/(60+1)
        assert len(result) == 1
        assert result[0]["rrf_score"] == pytest.approx(1.0 / 61)

    def test_rrf_preserves_vector_metadata(self, retriever):
        """RRF 병합 시 Vector 결과의 메타데이터가 유지됨"""
        vector_results = [{"text": "공통 텍스트", "metadata": {"source": "doc1.txt"}}]
        bm25_results = [{"text": "공통 텍스트", "metadata": {}}]
        result = retriever._rrf_fusion(vector_results, bm25_results)
        assert result[0]["metadata"] == {"source": "doc1.txt"}


class TestHybridRetrieverBM25:
    """BM25 검색 테스트"""

    def test_bm25_search_returns_list(self, retriever):
        """_bm25_search가 리스트를 반환"""
        result = retriever._bm25_search("파이썬", top_k=2)
        assert isinstance(result, list)

    def test_bm25_empty_corpus(self, mock_embedder):
        """빈 코퍼스에서 BM25 검색 시 빈 리스트"""
        vs = MagicMock()
        vs.count.return_value = 0
        retriever = HybridRetriever(vectorstore=vs, embedder=mock_embedder)
        result = retriever._bm25_search("쿼리", top_k=5)
        assert result == []

    def test_bm25_korean_query_produces_nonzero_scores(self, retriever):
        """한국어 쿼리로 검색 시 비영 BM25 점수 반환"""
        result = retriever._bm25_search("파이썬", top_k=2)
        assert len(result) > 0
        scores = [r["score"] for r in result]
        assert max(scores) > 0.0

    def test_bm25_korean_relevance_ranking(self, retriever):
        """관련 문서가 BM25 상위 랭킹"""
        result = retriever._bm25_search("파이썬", top_k=2)
        assert "파이썬" in result[0]["text"]

    def test_bm25_uses_pretokenized_data(self, retriever, mock_vectorstore):
        """사전 토큰화된 데이터를 사용하여 BM25 검색 수행"""
        retriever._bm25_search("파이썬", top_k=2)
        mock_vectorstore.get_all_texts_and_tokens.assert_called_once()

    def test_bm25_index_cached_on_repeated_search(self, retriever, mock_vectorstore):
        """연속 검색 시 BM25 인덱스가 캐시되어 재조회 안 함"""
        retriever._bm25_search("파이썬", top_k=2)
        retriever._bm25_search("머신러닝", top_k=2)
        # count()는 매번 호출되지만 get_all_texts_and_tokens()는 1번만
        assert mock_vectorstore.get_all_texts_and_tokens.call_count == 1

    def test_bm25_index_rebuilt_when_corpus_changes(self, retriever, mock_vectorstore):
        """코퍼스 크기 변경 시 BM25 인덱스 재구축"""
        retriever._bm25_search("파이썬", top_k=2)
        assert mock_vectorstore.get_all_texts_and_tokens.call_count == 1

        # 문서 추가 시뮬레이션
        mock_vectorstore.count.return_value = 5
        mock_vectorstore.get_all_texts_and_tokens.return_value = (
            [
                "파이썬은 인터프리터 언어입니다.",
                "머신러닝은 데이터 분석에 사용됩니다.",
                "자바스크립트는 웹 개발에 필수입니다.",
                "데이터베이스 설계는 중요합니다.",
                "새로운 문서가 추가되었습니다.",
            ],
            [
                ["파이썬", "인터프리터", "언어"],
                ["머신러닝", "데이터", "분석", "사용"],
                ["자바스크립트", "웹", "개발", "필수"],
                ["데이터베이스", "설계", "중요"],
                ["새로운", "문서", "추가"],
            ],
        )
        retriever._bm25_search("파이썬", top_k=2)
        assert mock_vectorstore.get_all_texts_and_tokens.call_count == 2

    def test_bm25_fallback_when_no_tokenized_data(self, mock_embedder):
        """토큰화 데이터가 없는 문서는 실시간 토큰화로 폴백"""
        vs = MagicMock()
        vs.count.return_value = 4
        vs.get_all_texts_and_tokens.return_value = (
            [
                "파이썬은 좋습니다.",
                "자바도 좋습니다.",
                "자바스크립트는 웹 언어입니다.",
                "데이터베이스 설계는 중요합니다.",
            ],
            [[], [], [], []],  # 토큰화 데이터 없음 → 실시간 폴백
        )
        retriever = HybridRetriever(vectorstore=vs, embedder=mock_embedder)
        result = retriever._bm25_search("파이썬", top_k=2)
        assert len(result) > 0
        scores = [r["score"] for r in result]
        assert max(scores) > 0.0

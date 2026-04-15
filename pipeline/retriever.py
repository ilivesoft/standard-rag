# 하이브리드 검색 모듈 - Vector + BM25 + RRF 융합 검색
from __future__ import annotations

from rank_bm25 import BM25Okapi

from pipeline.tokenizer import tokenize_korean
from pipeline.vectorstore_protocol import VectorStoreProtocol


class HybridRetriever:
    """Vector 검색과 BM25 검색을 RRF로 융합하는 하이브리드 검색기"""

    def __init__(self, vectorstore: VectorStoreProtocol, embedder):
        """초기화

        Args:
            vectorstore: VectorStoreProtocol 구현체 (ChromaVectorStore 또는 PGVectorStore)
            embedder: DocumentEmbedder 인스턴스
        """
        self._vectorstore = vectorstore
        self._embedder = embedder
        # BM25 인덱스 캐시
        self._bm25_index: BM25Okapi | None = None
        self._cached_texts: list[str] = []
        self._cached_count: int = -1

    # @MX:ANCHOR: [AUTO] 하이브리드 검색 공개 API - query_graph, 테스트 등 다수 호출
    # @MX:REASON: fan_in >= 3 (query_graph retrieve 노드, test_retriever, test_query_graph 등)
    def retrieve(self, query: str, top_k: int = 10, alpha: float = 0.5) -> list[dict]:
        """Vector + BM25 + RRF 하이브리드 검색을 수행합니다.

        Args:
            query: 검색 질의 문자열
            top_k: 반환할 최대 결과 수
            alpha: Vector 검색 가중치 (0=BM25, 1=Vector)

        Returns:
            RRF 점수로 정렬된 청크 리스트
        """
        query_embedding = self._embedder.embed_query(query)

        vector_results = self._vector_search(query_embedding, top_k)
        bm25_results = self._bm25_search(query, top_k)

        fused = self._rrf_fusion(vector_results, bm25_results, alpha=alpha)
        return fused[:top_k]

    def _vector_search(self, query_embedding: list[float], top_k: int) -> list[dict]:
        """벡터 유사도 검색을 수행합니다."""
        return self._vectorstore.search(query_embedding, top_k=top_k)

    def _build_bm25_index(self) -> bool:
        """BM25 인덱스를 구축하거나 캐시를 갱신합니다.

        인제스트 시 저장된 토큰화 데이터를 우선 사용하고,
        토큰화 데이터가 없는 문서는 실시간 토큰화로 폴백합니다.

        Returns:
            인덱스가 유효하면 True, 문서가 없으면 False
        """
        current_count = self._vectorstore.count()
        if current_count == 0:
            self._bm25_index = None
            self._cached_texts = []
            self._cached_count = 0
            return False

        # 코퍼스 크기가 변경된 경우에만 재구축
        if current_count != self._cached_count:
            self._cached_texts, tokenized_corpus = (
                self._vectorstore.get_all_texts_and_tokens()
            )
            # 토큰화 데이터가 없는 문서는 실시간 토큰화로 폴백
            for i, tokens in enumerate(tokenized_corpus):
                if not tokens:
                    tokenized_corpus[i] = tokenize_korean(self._cached_texts[i])
            self._bm25_index = BM25Okapi(tokenized_corpus)
            self._cached_count = current_count

        return True

    def _bm25_search(self, query: str, top_k: int) -> list[dict]:
        """BM25 기반 키워드 검색을 수행합니다."""
        if not self._build_bm25_index():
            return []

        tokenized_query = tokenize_korean(query)
        scores = self._bm25_index.get_scores(tokenized_query)  # type: ignore[union-attr]

        # 점수 내림차순으로 인덱스 정렬
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        top_indices = ranked_indices[:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "text": self._cached_texts[idx],
                "metadata": {},
                "score": float(scores[idx]),
            })
        return results

    # @MX:NOTE: [AUTO] RRF k=60은 Cormack et al.(2009) 논문 권장값 - 임의 변경 금지
    def _rrf_fusion(
        self,
        vector_results: list[dict],
        bm25_results: list[dict],
        k: int = 60,
        alpha: float = 0.5,
    ) -> list[dict]:
        """Reciprocal Rank Fusion으로 두 검색 결과를 병합합니다.

        Args:
            vector_results: 벡터 검색 결과 (순위별 정렬됨)
            bm25_results: BM25 검색 결과 (순위별 정렬됨)
            k: RRF 하이퍼파라미터 (일반적으로 60)
            alpha: Vector 검색 가중치 (0=BM25만, 1=Vector만, 0.5=동일 가중치)

        Returns:
            RRF 점수로 정렬된 병합 결과
        """
        if not vector_results and not bm25_results:
            return []

        # 텍스트를 키로 RRF 점수 계산 (alpha 가중치 적용)
        rrf_scores: dict[str, float] = {}
        text_to_chunk: dict[str, dict] = {}

        for rank, chunk in enumerate(vector_results):
            text = chunk["text"]
            rrf_scores[text] = rrf_scores.get(text, 0) + alpha / (k + rank + 1)
            text_to_chunk[text] = chunk

        for rank, chunk in enumerate(bm25_results):
            text = chunk["text"]
            rrf_scores[text] = rrf_scores.get(text, 0) + (1 - alpha) / (k + rank + 1)
            if text not in text_to_chunk:
                text_to_chunk[text] = chunk

        # RRF 점수 내림차순 정렬
        sorted_texts = sorted(rrf_scores.keys(), key=lambda t: rrf_scores[t], reverse=True)

        result = []
        for text in sorted_texts:
            chunk = dict(text_to_chunk[text])
            chunk["rrf_score"] = rrf_scores[text]
            result.append(chunk)
        return result

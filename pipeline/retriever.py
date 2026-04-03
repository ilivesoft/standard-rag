# 하이브리드 검색 모듈 - Vector + BM25 + RRF 융합 검색
from rank_bm25 import BM25Okapi


class HybridRetriever:
    """Vector 검색과 BM25 검색을 RRF로 융합하는 하이브리드 검색기"""

    def __init__(self, vectorstore, embedder):
        """초기화

        Args:
            vectorstore: VectorStore 인스턴스
            embedder: DocumentEmbedder 인스턴스
        """
        self._vectorstore = vectorstore
        self._embedder = embedder

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

        fused = self._rrf_fusion(vector_results, bm25_results)
        return fused[:top_k]

    def _vector_search(self, query_embedding: list[float], top_k: int) -> list[dict]:
        """벡터 유사도 검색을 수행합니다."""
        return self._vectorstore.search(query_embedding, top_k=top_k)

    def _bm25_search(self, query: str, top_k: int) -> list[dict]:
        """BM25 기반 키워드 검색을 수행합니다."""
        all_texts = self._vectorstore.get_all_texts()
        if not all_texts:
            return []

        # 텍스트를 토큰화하여 BM25 인덱스 구성
        tokenized_corpus = [text.split() for text in all_texts]
        bm25 = BM25Okapi(tokenized_corpus)

        tokenized_query = query.split()
        scores = bm25.get_scores(tokenized_query)

        # 점수 내림차순으로 인덱스 정렬
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        top_indices = ranked_indices[:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "text": all_texts[idx],
                "metadata": {},
                "score": float(scores[idx]),
            })
        return results

    # @MX:NOTE: [AUTO] RRF k=60은 Cormack et al.(2009) 논문 권장값 - 임의 변경 금지
    def _rrf_fusion(self, vector_results: list[dict], bm25_results: list[dict], k: int = 60) -> list[dict]:
        """Reciprocal Rank Fusion으로 두 검색 결과를 병합합니다.

        Args:
            vector_results: 벡터 검색 결과 (순위별 정렬됨)
            bm25_results: BM25 검색 결과 (순위별 정렬됨)
            k: RRF 하이퍼파라미터 (일반적으로 60)

        Returns:
            RRF 점수로 정렬된 병합 결과
        """
        if not vector_results and not bm25_results:
            return []

        # 텍스트를 키로 RRF 점수 계산
        rrf_scores: dict[str, float] = {}
        text_to_chunk: dict[str, dict] = {}

        for rank, chunk in enumerate(vector_results):
            text = chunk["text"]
            rrf_scores[text] = rrf_scores.get(text, 0) + 1.0 / (k + rank + 1)
            text_to_chunk[text] = chunk

        for rank, chunk in enumerate(bm25_results):
            text = chunk["text"]
            rrf_scores[text] = rrf_scores.get(text, 0) + 1.0 / (k + rank + 1)
            text_to_chunk[text] = chunk

        # RRF 점수 내림차순 정렬
        sorted_texts = sorted(rrf_scores.keys(), key=lambda t: rrf_scores[t], reverse=True)

        result = []
        for text in sorted_texts:
            chunk = dict(text_to_chunk[text])
            chunk["rrf_score"] = rrf_scores[text]
            result.append(chunk)
        return result

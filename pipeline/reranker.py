# Reranker 모듈 - cross-encoder 기반 문서 재순위화
# Lazy import: 실제 사용 시 로드
try:
    from sentence_transformers.cross_encoder import CrossEncoder
except ImportError:
    CrossEncoder = None  # type: ignore


class DocumentReranker:
    """CrossEncoder를 사용하여 검색 결과를 재순위화하는 클래스 (Lazy Loading 지원)"""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """초기화 (모델은 실제 사용 시 로드됨)

        Args:
            model_name: 사용할 CrossEncoder 모델 이름
        """
        self._model_name = model_name
        self._model = None  # 지연 로딩

    def _load_model(self) -> None:
        """모델을 지연 로딩합니다."""
        if self._model is None:
            self._model = CrossEncoder(self._model_name)

    def rerank(self, query: str, chunks: list[dict], top_n: int = 3) -> list[dict]:
        """(query, document) 쌍을 재점수화하여 상위 N개를 반환합니다.

        Args:
            query: 검색 질의
            chunks: 재순위화할 청크 리스트
            top_n: 반환할 최대 청크 수

        Returns:
            재순위화된 상위 N개 청크 (rerank_score 포함)
        """
        if not chunks:
            return []

        self._load_model()
        assert self._model is not None

        pairs = [(query, chunk["text"]) for chunk in chunks]
        scores = self._model.predict(pairs)

        scored_chunks = []
        for chunk, score in zip(chunks, scores):
            chunk_copy = dict(chunk)
            chunk_copy["rerank_score"] = float(score)
            scored_chunks.append(chunk_copy)

        # 점수 내림차순 정렬 후 상위 N개 반환
        scored_chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored_chunks[:top_n]

    @property
    def is_loaded(self) -> bool:
        """모델이 로드되었는지 확인합니다."""
        return self._model is not None

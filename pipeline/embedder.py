# 임베딩 생성 모듈 - BAAI/bge-m3, SentenceTransformer 기반
from __future__ import annotations

# Lazy import: 실제 사용 시 로드
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None  # type: ignore


class DocumentEmbedder:
    """문서와 질의를 위한 임베딩 벡터를 생성하는 클래스 (Lazy Loading 지원)"""

    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = "cpu"):
        """초기화 (모델은 실제 사용 시 로드됨)

        Args:
            model_name: 사용할 SentenceTransformer 모델 이름
            device: 연산 장치 ('cpu' 또는 'cuda')
        """
        self._model_name = model_name
        self._device = device
        self._model = None  # 지연 로딩

    def _load_model(self) -> None:
        """모델을 지연 로딩합니다."""
        if self._model is None:
            self._model = SentenceTransformer(self._model_name, device=self._device)

    # @MX:ANCHOR: [AUTO] 배치 임베딩 공개 API - ingest_graph, 테스트 등 다수 호출
    # @MX:REASON: fan_in >= 3 (ingest_graph embed 노드, test_embedder, test_ingest_graph 등)
    def embed(self, texts: list[str]) -> list[list[float]]:
        """배치 임베딩을 생성합니다.

        Args:
            texts: 임베딩할 텍스트 리스트

        Returns:
            각 텍스트의 임베딩 벡터 리스트
        """
        if not texts:
            return []

        self._load_model()
        assert self._model is not None
        vectors = self._model.encode(texts, convert_to_numpy=True)

        if len(vectors.shape) == 1:
            # 단일 벡터인 경우 2D로 변환
            return [vectors.tolist()]
        return [v.tolist() for v in vectors]

    def embed_query(self, query: str) -> list[float]:
        """단일 질의에 대한 임베딩을 생성합니다.

        Args:
            query: 임베딩할 질의 문자열

        Returns:
            질의 임베딩 벡터
        """
        self._load_model()
        assert self._model is not None
        vector = self._model.encode(query, convert_to_numpy=True)
        return vector.tolist()

    @property
    def is_loaded(self) -> bool:
        """모델이 로드되었는지 확인합니다."""
        return self._model is not None

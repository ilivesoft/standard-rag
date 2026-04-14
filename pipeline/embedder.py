# 임베딩 생성 모듈 - BAAI/bge-m3, SentenceTransformer 기반 (LangChain Embeddings 인터페이스 구현)
from __future__ import annotations

from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer


class DocumentEmbedder(Embeddings):
    """문서와 질의를 위한 임베딩 벡터를 생성하는 클래스 (Lazy Loading 지원).

    LangChain Embeddings 인터페이스를 구현하여 langchain_chroma.Chroma의
    embedding_function으로 직접 사용 가능합니다.
    """

    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = "cpu"):
        """초기화 (모델은 실제 사용 시 로드됨)

        Args:
            model_name: 사용할 SentenceTransformer 모델 이름
            device: 연산 장치 ('cpu' 또는 'cuda')
        """
        self._model_name = model_name
        self._device = device
        self._model: SentenceTransformer | None = None

    def _load_model(self) -> SentenceTransformer:
        """모델을 지연 로딩합니다."""
        if self._model is None:
            self._model = SentenceTransformer(self._model_name, device=self._device)
        return self._model

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

        vectors = self._load_model().encode(texts, convert_to_numpy=True)

        if len(vectors.shape) == 1:
            return [vectors.tolist()]
        return [v.tolist() for v in vectors]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """LangChain Embeddings 인터페이스 구현 - 배치 임베딩.

        langchain_chroma.Chroma의 embedding_function으로 사용됩니다.
        """
        return self.embed(texts)

    def embed_query(self, text: str) -> list[float]:
        """단일 질의에 대한 임베딩을 생성합니다.

        Args:
            text: 임베딩할 질의 문자열

        Returns:
            질의 임베딩 벡터
        """
        return self._load_model().encode(text, convert_to_numpy=True).tolist()

    @property
    def is_loaded(self) -> bool:
        """모델이 로드되었는지 확인합니다."""
        return self._model is not None

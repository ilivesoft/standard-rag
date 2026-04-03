# 청킹 모듈 - SentenceSplitter 기반 텍스트 분할
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document


class TextChunker:
    """텍스트를 지정한 크기의 청크로 분할하는 클래스"""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        """초기화

        Args:
            chunk_size: 각 청크의 최대 토큰 수
            chunk_overlap: 연속된 청크 간 중복 토큰 수
        """
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            paragraph_separator="\n\n",
        )

    def chunk(self, text: str, metadata: dict) -> list[dict]:
        """텍스트를 청크로 분할하고 메타데이터를 첨부합니다.

        Args:
            text: 분할할 텍스트
            metadata: 각 청크에 첨부할 기본 메타데이터 (source 등)

        Returns:
            청크 딕셔너리 리스트. 각 항목:
            {"text": str, "metadata": {"source": str, "chunk_index": int, ...}}
        """
        if not text or not text.strip():
            return []

        nodes = self._splitter.get_nodes_from_documents([Document(text=text)])

        result = []
        for i, node in enumerate(nodes):
            if node.text.strip():
                chunk_metadata = {**metadata, "chunk_index": i}
                result.append({"text": node.text, "metadata": chunk_metadata})

        return result

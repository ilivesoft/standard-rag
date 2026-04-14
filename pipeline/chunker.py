# 청킹 모듈 - RecursiveCharacterTextSplitter 기반 텍스트 분할
from langchain_text_splitters import RecursiveCharacterTextSplitter


class TextChunker:
    """텍스트를 지정한 크기의 청크로 분할하는 클래스"""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        """초기화

        Args:
            chunk_size: 각 청크의 최대 문자 수
            chunk_overlap: 연속된 청크 간 중복 문자 수
        """
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
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

        docs = self._splitter.create_documents([text], metadatas=[metadata])
        return [
            {"text": doc.page_content, "metadata": {**doc.metadata, "chunk_index": i}}
            for i, doc in enumerate(docs)
            if doc.page_content.strip()
        ]

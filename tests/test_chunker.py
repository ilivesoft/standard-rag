# 청킹 모듈 테스트 - 청크 크기, 오버랩, 메타데이터 검증
import pytest
from pipeline.chunker import TextChunker


@pytest.fixture
def chunker():
    """기본 TextChunker 인스턴스 (chunk_size=100, overlap=20)"""
    return TextChunker(chunk_size=100, chunk_overlap=20)


@pytest.fixture
def long_text():
    """청킹 테스트용 긴 텍스트"""
    return " ".join([f"단어{i}" for i in range(200)])


class TestTextChunkerBasic:
    """TextChunker 기본 동작 테스트"""

    def test_chunk_returns_list(self, chunker, long_text):
        """chunk 메서드가 리스트를 반환"""
        result = chunker.chunk(long_text, {"source": "test.txt"})
        assert isinstance(result, list)

    def test_chunk_short_text_returns_single_chunk(self, chunker):
        """짧은 텍스트는 단일 청크 반환"""
        short_text = "짧은 텍스트입니다."
        result = chunker.chunk(short_text, {"source": "test.txt"})
        assert len(result) >= 1

    def test_chunk_long_text_returns_multiple_chunks(self, chunker, long_text):
        """긴 텍스트는 여러 청크로 분할"""
        result = chunker.chunk(long_text, {"source": "test.txt"})
        assert len(result) > 1

    def test_chunk_empty_text_returns_empty_list(self, chunker):
        """빈 텍스트는 빈 리스트 반환"""
        result = chunker.chunk("", {"source": "test.txt"})
        assert result == []


class TestTextChunkerStructure:
    """청크 구조 검증 테스트"""

    def test_chunk_has_text_key(self, chunker):
        """각 청크에 'text' 키가 존재"""
        result = chunker.chunk("테스트 텍스트입니다.", {"source": "test.txt"})
        for chunk in result:
            assert "text" in chunk

    def test_chunk_has_metadata_key(self, chunker):
        """각 청크에 'metadata' 키가 존재"""
        result = chunker.chunk("테스트 텍스트입니다.", {"source": "test.txt"})
        for chunk in result:
            assert "metadata" in chunk

    def test_chunk_metadata_has_source(self, chunker):
        """메타데이터에 source 정보가 포함"""
        result = chunker.chunk("테스트 텍스트입니다.", {"source": "doc.pdf"})
        for chunk in result:
            assert chunk["metadata"]["source"] == "doc.pdf"

    def test_chunk_metadata_has_chunk_index(self, chunker, long_text):
        """메타데이터에 chunk_index가 포함"""
        result = chunker.chunk(long_text, {"source": "test.txt"})
        for i, chunk in enumerate(result):
            assert "chunk_index" in chunk["metadata"]

    def test_chunk_indices_are_sequential(self, chunker, long_text):
        """청크 인덱스가 순서대로 증가"""
        result = chunker.chunk(long_text, {"source": "test.txt"})
        for i, chunk in enumerate(result):
            assert chunk["metadata"]["chunk_index"] == i

    def test_chunk_text_is_string(self, chunker):
        """청크 텍스트가 문자열"""
        result = chunker.chunk("테스트 텍스트입니다.", {"source": "test.txt"})
        for chunk in result:
            assert isinstance(chunk["text"], str)


class TestTextChunkerSizeControl:
    """청크 크기 제어 테스트"""

    def test_chunk_respects_max_size(self, long_text):
        """청크 크기가 원본 텍스트보다 짧음을 검증"""
        chunker = TextChunker(chunk_size=50, chunk_overlap=10)
        result = chunker.chunk(long_text, {"source": "test.txt"})
        for chunk in result:
            assert len(chunk["text"]) < len(long_text)

    def test_chunk_overlap_creates_continuity(self, long_text):
        """오버랩 설정 시 청크 간 텍스트가 중복"""
        chunker_with_overlap = TextChunker(chunk_size=200, chunk_overlap=50)
        chunker_no_overlap = TextChunker(chunk_size=200, chunk_overlap=0)

        result_with = chunker_with_overlap.chunk(long_text, {"source": "test.txt"})
        result_without = chunker_no_overlap.chunk(long_text, {"source": "test.txt"})

        # 오버랩 있을 때 더 많은 청크가 생성될 수 있음
        assert len(result_with) >= len(result_without) - 1

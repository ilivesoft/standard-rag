# 벡터 저장소 관리 메서드 테스트 - 문서 조회, 삭제 CRUD 검증
import pytest
import numpy as np
from pipeline.vectorstore import VectorStore


@pytest.fixture
def vector_store(tmp_path):
    """tmp_path를 사용하는 격리된 VectorStore 인스턴스"""
    return VectorStore(
        persist_dir=str(tmp_path / "chroma"),
        collection="test_collection",
    )


@pytest.fixture
def multi_doc_chunks():
    """여러 문서의 샘플 청크"""
    return [
        {"text": "문서1 첫 번째 청크", "metadata": {"source": "doc1.pdf", "chunk_index": 0}},
        {"text": "문서1 두 번째 청크", "metadata": {"source": "doc1.pdf", "chunk_index": 1}},
        {"text": "문서2 첫 번째 청크", "metadata": {"source": "doc2.txt", "chunk_index": 0}},
        {"text": "문서2 두 번째 청크", "metadata": {"source": "doc2.txt", "chunk_index": 1}},
        {"text": "문서2 세 번째 청크", "metadata": {"source": "doc2.txt", "chunk_index": 2}},
    ]


@pytest.fixture
def multi_doc_embeddings():
    """여러 문서용 임베딩"""
    np.random.seed(42)
    return [np.random.rand(384).tolist() for _ in range(5)]


class TestGetAllDocuments:
    """get_all_documents 메서드 테스트"""

    def test_empty_store_returns_empty(self, vector_store):
        """빈 저장소에서 빈 목록 반환"""
        result = vector_store.get_all_documents()
        assert result == []

    def test_returns_grouped_by_source(self, vector_store, multi_doc_chunks, multi_doc_embeddings):
        """소스별로 그룹화된 문서 목록 반환"""
        vector_store.add(multi_doc_chunks, multi_doc_embeddings)
        result = vector_store.get_all_documents()
        assert len(result) == 2

        sources = {doc["source"] for doc in result}
        assert sources == {"doc1.pdf", "doc2.txt"}

    def test_chunk_count_per_document(self, vector_store, multi_doc_chunks, multi_doc_embeddings):
        """문서별 청크 수가 정확"""
        vector_store.add(multi_doc_chunks, multi_doc_embeddings)
        result = vector_store.get_all_documents()

        doc_map = {doc["source"]: doc["chunk_count"] for doc in result}
        assert doc_map["doc1.pdf"] == 2
        assert doc_map["doc2.txt"] == 3


class TestGetBySource:
    """get_by_source 메서드 테스트"""

    def test_returns_chunks_for_source(self, vector_store, multi_doc_chunks, multi_doc_embeddings):
        """특정 소스의 청크만 반환"""
        vector_store.add(multi_doc_chunks, multi_doc_embeddings)
        result = vector_store.get_by_source("doc1.pdf")
        assert len(result) == 2
        for chunk in result:
            assert chunk["metadata"]["source"] == "doc1.pdf"

    def test_nonexistent_source_returns_empty(self, vector_store, multi_doc_chunks, multi_doc_embeddings):
        """존재하지 않는 소스는 빈 리스트 반환"""
        vector_store.add(multi_doc_chunks, multi_doc_embeddings)
        result = vector_store.get_by_source("nonexistent.txt")
        assert result == []

    def test_result_has_chunk_id_and_text(self, vector_store, multi_doc_chunks, multi_doc_embeddings):
        """결과에 chunk_id와 text 포함"""
        vector_store.add(multi_doc_chunks, multi_doc_embeddings)
        result = vector_store.get_by_source("doc1.pdf")
        for chunk in result:
            assert "chunk_id" in chunk
            assert "text" in chunk
            assert "metadata" in chunk


class TestDeleteBySource:
    """delete_by_source 메서드 테스트"""

    def test_deletes_correct_documents(self, vector_store, multi_doc_chunks, multi_doc_embeddings):
        """특정 소스의 문서만 삭제"""
        vector_store.add(multi_doc_chunks, multi_doc_embeddings)
        assert vector_store.count() == 5

        deleted = vector_store.delete_by_source("doc1.pdf")
        assert deleted == 2
        assert vector_store.count() == 3

    def test_other_documents_preserved(self, vector_store, multi_doc_chunks, multi_doc_embeddings):
        """삭제 후 다른 문서는 유지"""
        vector_store.add(multi_doc_chunks, multi_doc_embeddings)
        vector_store.delete_by_source("doc1.pdf")

        remaining = vector_store.get_by_source("doc2.txt")
        assert len(remaining) == 3

    def test_delete_nonexistent_returns_zero(self, vector_store, multi_doc_chunks, multi_doc_embeddings):
        """존재하지 않는 소스 삭제 시 0 반환"""
        vector_store.add(multi_doc_chunks, multi_doc_embeddings)
        deleted = vector_store.delete_by_source("nonexistent.txt")
        assert deleted == 0


class TestDeleteAll:
    """delete_all 메서드 테스트"""

    def test_deletes_all_chunks(self, vector_store, multi_doc_chunks, multi_doc_embeddings):
        """모든 청크 삭제"""
        vector_store.add(multi_doc_chunks, multi_doc_embeddings)
        assert vector_store.count() == 5

        deleted = vector_store.delete_all()
        assert deleted == 5
        assert vector_store.count() == 0

    def test_delete_all_empty_store(self, vector_store):
        """빈 저장소에서 delete_all은 0 반환"""
        deleted = vector_store.delete_all()
        assert deleted == 0

    def test_can_add_after_delete_all(self, vector_store, multi_doc_chunks, multi_doc_embeddings):
        """delete_all 후에도 다시 추가 가능"""
        vector_store.add(multi_doc_chunks, multi_doc_embeddings)
        vector_store.delete_all()

        new_chunks = [{"text": "새 청크", "metadata": {"source": "new.txt", "chunk_index": 0}}]
        new_embeddings = [np.random.rand(384).tolist()]
        vector_store.add(new_chunks, new_embeddings)
        assert vector_store.count() == 1

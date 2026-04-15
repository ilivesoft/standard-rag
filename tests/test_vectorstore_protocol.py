# VectorStoreProtocol 테스트 - 프로토콜 계약 및 ChromaVectorStore 호환성 검증
from pipeline.vectorstore_protocol import VectorStoreProtocol
from pipeline.vectorstore_chromavector import ChromaVectorStore


class TestVectorStoreProtocolDefinition:
    """VectorStoreProtocol이 올바르게 정의되어 있는지 검증합니다."""

    def test_protocol_is_runtime_checkable(self):
        """VectorStoreProtocol이 runtime_checkable이어야 합니다."""
        # isinstance 호출 자체가 가능한지 확인 (runtime_checkable 필요)
        result = isinstance(object(), VectorStoreProtocol)
        assert isinstance(result, bool)

    def test_protocol_has_add_method(self):
        """add 메서드가 프로토콜에 정의되어 있어야 합니다."""
        assert hasattr(VectorStoreProtocol, "add")

    def test_protocol_has_search_method(self):
        """search 메서드가 프로토콜에 정의되어 있어야 합니다."""
        assert hasattr(VectorStoreProtocol, "search")

    def test_protocol_has_count_method(self):
        """count 메서드가 프로토콜에 정의되어 있어야 합니다."""
        assert hasattr(VectorStoreProtocol, "count")

    def test_protocol_has_get_all_texts_method(self):
        """get_all_texts 메서드가 프로토콜에 정의되어 있어야 합니다."""
        assert hasattr(VectorStoreProtocol, "get_all_texts")

    def test_protocol_has_get_all_texts_and_tokens_method(self):
        """get_all_texts_and_tokens 메서드가 프로토콜에 정의되어 있어야 합니다."""
        assert hasattr(VectorStoreProtocol, "get_all_texts_and_tokens")

    def test_protocol_has_list_collections_method(self):
        """list_collections 메서드가 프로토콜에 정의되어 있어야 합니다."""
        assert hasattr(VectorStoreProtocol, "list_collections")

    def test_protocol_has_get_all_documents_method(self):
        """get_all_documents 메서드가 프로토콜에 정의되어 있어야 합니다."""
        assert hasattr(VectorStoreProtocol, "get_all_documents")

    def test_protocol_has_get_by_source_method(self):
        """get_by_source 메서드가 프로토콜에 정의되어 있어야 합니다."""
        assert hasattr(VectorStoreProtocol, "get_by_source")

    def test_protocol_has_delete_by_source_method(self):
        """delete_by_source 메서드가 프로토콜에 정의되어 있어야 합니다."""
        assert hasattr(VectorStoreProtocol, "delete_by_source")

    def test_protocol_has_delete_all_method(self):
        """delete_all 메서드가 프로토콜에 정의되어 있어야 합니다."""
        assert hasattr(VectorStoreProtocol, "delete_all")


class TestChromaVectorStoreProtocolCompliance:
    """ChromaVectorStore가 VectorStoreProtocol을 만족하는지 검증합니다."""

    def test_chroma_vectorstore_is_instance_of_protocol(self, tmp_path):
        """ChromaVectorStore 인스턴스가 VectorStoreProtocol isinstance 체크를 통과해야 합니다."""
        store = ChromaVectorStore(
            persist_dir=str(tmp_path / "chroma"),
            collection="test_protocol",
        )
        assert isinstance(store, VectorStoreProtocol)

    def test_chroma_vectorstore_class_satisfies_protocol(self):
        """ChromaVectorStore 클래스 자체가 프로토콜의 모든 메서드를 가져야 합니다."""
        required_methods = [
            "add", "search", "count", "get_all_texts",
            "get_all_texts_and_tokens", "list_collections",
            "get_all_documents", "get_by_source",
            "delete_by_source", "delete_all",
        ]
        for method_name in required_methods:
            assert hasattr(ChromaVectorStore, method_name), (
                f"ChromaVectorStore에 '{method_name}' 메서드가 없습니다."
            )

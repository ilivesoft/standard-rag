# ChromaDB 벡터 저장소 모듈 - 임베딩 저장 및 유사도 검색
import uuid
import chromadb


class ChromaVectorStore:
    """ChromaDB PersistentClient 기반 벡터 저장소 구현체.

    VectorStoreProtocol을 구현하며 로컬 파일 기반 ChromaDB를 사용합니다.
    개발 환경 기본 백엔드로 사용됩니다.
    """

    def __init__(self, persist_dir: str = "./storage/chroma", collection: str = "default"):
        """초기화

        Args:
            persist_dir: ChromaDB 저장 경로
            collection: 사용할 컬렉션 이름
        """
        self._persist_dir = persist_dir
        self._collection_name = collection
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": "cosine"},
        )

    # @MX:ANCHOR: [AUTO] 청크/임베딩 저장 공개 API - ingest_graph, 테스트 등 다수 호출
    # @MX:REASON: fan_in >= 3 (ingest_graph, test_vectorstore, test_api 등)
    def add(self, chunks: list[dict], embeddings: list[list[float]]) -> int:
        """청크와 임베딩을 저장합니다.

        Args:
            chunks: 저장할 청크 리스트 (text, metadata 포함)
            embeddings: 각 청크에 대응하는 임베딩 벡터 리스트

        Returns:
            저장된 청크 수
        """
        if not chunks:
            return 0

        ids = [str(uuid.uuid4()) for _ in chunks]
        texts = [c["text"] for c in chunks]
        metadatas = [c.get("metadata", {}) for c in chunks]

        self._collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,  # type: ignore[arg-type]
            metadatas=metadatas,
        )
        return len(chunks)

    # @MX:ANCHOR: [AUTO] 벡터 검색 공개 API - retriever, health check, 테스트 등 다수 호출
    # @MX:REASON: fan_in >= 3 (HybridRetriever._vector_search, health_check, test_vectorstore 등)
    def search(self, query_embedding: list[float], top_k: int = 10) -> list[dict]:
        """벡터 유사도 검색을 수행합니다.

        Args:
            query_embedding: 질의 임베딩 벡터
            top_k: 반환할 최대 결과 수

        Returns:
            유사도 순으로 정렬된 청크 딕셔너리 리스트
        """
        count = self.count()
        if count == 0:
            return []

        n_results = min(top_k, count)
        results = self._collection.query(
            query_embeddings=[query_embedding],  # type: ignore[arg-type]
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        output = []
        documents = (results["documents"] or [[]])[0]
        metadatas = (results["metadatas"] or [[]])[0]
        distances = (results["distances"] or [[]])[0]

        for doc, meta, dist in zip(documents, metadatas, distances):
            output.append({
                "text": doc,
                "metadata": meta or {},
                "score": float(1 - dist),  # 코사인 거리를 유사도로 변환
            })
        return output

    def get_all_texts(self) -> list[str]:
        """BM25 검색을 위해 저장된 모든 텍스트를 반환합니다."""
        count = self.count()
        if count == 0:
            return []

        results = self._collection.get(include=["documents"])
        return results["documents"] or []

    def get_all_texts_and_tokens(self) -> tuple[list[str], list[list[str]]]:
        """모든 텍스트와 사전 토큰화된 데이터를 반환합니다.

        Returns:
            (원본 텍스트 리스트, 토큰 리스트의 리스트) 튜플.
            토큰화 데이터가 없는 문서는 빈 리스트로 반환됩니다.
        """
        count = self.count()
        if count == 0:
            return [], []

        results = self._collection.get(include=["documents", "metadatas"])
        documents = results["documents"] or []
        metadatas = results["metadatas"] or []

        tokenized = []
        for meta in metadatas:
            token_str = (meta or {}).get("tokenized_text", "")
            tokenized.append(token_str.split() if token_str else [])

        return documents, tokenized

    # @MX:ANCHOR: [AUTO] 컬렉션 크기 확인 - query_graph, search, health check 등 다수 호출
    # @MX:REASON: fan_in >= 3 (query_graph empty check, search 내부, api/query health 등)
    def count(self) -> int:
        """저장된 청크의 총 수를 반환합니다."""
        return self._collection.count()

    def list_collections(self) -> list[str]:
        """사용 가능한 컬렉션 목록을 반환합니다."""
        return [c.name for c in self._client.list_collections()]

    def get_all_documents(self) -> list[dict]:
        """저장된 모든 문서를 소스 파일 단위로 그룹화하여 반환합니다.

        Returns:
            소스 파일별 문서 리스트 [{"source": "file.pdf", "chunk_count": 5}, ...]
        """
        results = self._collection.get(include=["metadatas"])
        metadatas = results.get("metadatas") or []

        source_counts: dict[str, int] = {}
        for meta in metadatas:
            source = str((meta or {}).get("source", "unknown"))
            source_counts[source] = source_counts.get(source, 0) + 1

        return [{"source": src, "chunk_count": cnt} for src, cnt in source_counts.items()]

    def get_by_source(self, source: str) -> list[dict]:
        """특정 소스 문서의 모든 청크를 반환합니다.

        Args:
            source: 조회할 소스 파일명

        Returns:
            해당 소스의 청크 리스트 [{"chunk_id": "uuid", "text": "...", "metadata": {...}}, ...]
        """
        results = self._collection.get(
            where={"source": source},
            include=["documents", "metadatas"],
        )

        ids = results.get("ids") or []
        documents = results.get("documents") or []
        metadatas = results.get("metadatas") or []

        return [
            {"chunk_id": cid, "text": doc, "metadata": meta or {}}
            for cid, doc, meta in zip(ids, documents, metadatas)
        ]

    def delete_by_source(self, source: str) -> int:
        """특정 소스 문서의 모든 청크를 삭제합니다.

        Args:
            source: 삭제할 소스 파일명

        Returns:
            삭제된 청크 수
        """
        results = self._collection.get(where={"source": source})
        ids = results.get("ids") or []

        if not ids:
            return 0

        self._collection.delete(ids=ids)
        return len(ids)

    def delete_all(self) -> int:
        """저장된 모든 데이터를 삭제합니다.

        컬렉션을 삭제하고 동일한 설정으로 재생성합니다.

        Returns:
            삭제된 청크 수
        """
        deleted_count = self.count()

        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        return deleted_count

# 삭제 예정
# # 하위 호환성을 위한 alias - 기존 코드가 VectorStore를 직접 참조하는 경우 유지
# VectorStore = ChromaVectorStore

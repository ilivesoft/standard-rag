# PGVector 벡터 저장소 구현체 - PostgreSQL + pgvector 기반 운영 환경 백엔드
# @MX:NOTE: [AUTO] langchain-postgres 외부 의존 모듈. PGVector 래퍼 사용.
#            langchain_pg_embedding, langchain_pg_collection 테이블을 내부적으로 관리함.
# @MX:WARN: [AUTO] 생성자에서 Fail-Fast 연결 검증 수행. 연결 실패 시 즉시 예외 발생.
# @MX:REASON: 운영 환경에서 연결 실패를 빠르게 감지하여 장애를 조기에 차단하기 위함
import logging

from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)


class PGVectorStore:
    """PostgreSQL + pgvector 기반 벡터 저장소 구현체.

    VectorStoreProtocol을 완전히 구현하며 운영 환경의 기본 백엔드로 사용됩니다.
    내부적으로 langchain_postgres.PGVector를 활용합니다.

    연결 정보(비밀번호 등)는 로그에 절대 출력되지 않습니다.
    """

    def __init__(
        self,
        host: str,
        port: int,
        database: str,
        user: str,
        password: str,
        collection: str,
        dimension: int,
    ) -> None:
        """PGVectorStore를 초기화하고 PostgreSQL 연결을 검증합니다.

        Args:
            host: PostgreSQL 호스트 주소.
            port: PostgreSQL 포트 번호.
            database: 데이터베이스 이름.
            user: 접속 사용자명.
            password: 접속 비밀번호 (로그에 노출되지 않음).
            collection: 컬렉션(테이블) 이름.
            dimension: 임베딩 벡터 차원 수.

        Raises:
            ConnectionError: PostgreSQL 연결에 실패한 경우 (Fail-Fast).
        """
        self._host = host
        self._port = port
        self._database = database
        self._user = user
        self._collection_name = collection
        self._dimension = dimension

        # 연결 문자열 생성 (비밀번호는 내부에서만 사용)
        connection_string = self._build_connection_string(
            host=host, port=port, database=database, user=user, password=password
        )

        # Fail-Fast: 생성자에서 연결 검증
        self._validate_connection(connection_string)

        # PGVector 스토어 초기화 (FakeEmbeddings 사용 - 임베딩은 직접 제공)
        from langchain_postgres.vectorstores import PGVector as LCPGVector

        self._store = LCPGVector(
            embeddings=_NoOpEmbeddings(dimension),
            collection_name=collection,
            connection=connection_string,
            use_jsonb=True,
        )

        logger.info(
            "PGVectorStore 초기화 완료 - 호스트: %s, 포트: %d, DB: %s, 컬렉션: %s",
            host, port, database, collection
        )

    @staticmethod
    def _build_connection_string(
        host: str,
        port: int,
        database: str,
        user: str,
        password: str,
    ) -> str:
        """SQLAlchemy 연결 문자열을 생성합니다.

        Args:
            host: 호스트 주소.
            port: 포트 번호.
            database: 데이터베이스 이름.
            user: 사용자명.
            password: 비밀번호.

        Returns:
            postgresql+psycopg:// 형식의 연결 문자열.
        """
        return f"postgresql+psycopg://{user}:{password}@{host}:{port}/{database}"

    def _validate_connection(self, connection_string: str) -> None:
        """PostgreSQL 연결 가능 여부를 검증합니다.

        Args:
            connection_string: 검증할 연결 문자열.

        Raises:
            ConnectionError: 연결 실패 시 (비밀번호는 메시지에 포함되지 않음).
        """
        try:
            engine = create_engine(connection_string)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            engine.dispose()
        except Exception as e:
            # 비밀번호가 메시지에 노출되지 않도록 호스트/포트/DB 정보만 포함
            raise ConnectionError(
                f"PostgreSQL 연결 실패 - 호스트: {self._host}, "
                f"포트: {self._port}, DB: {self._database}. "
                f"원인: {type(e).__name__}"
            ) from e

    def add(self, chunks: list[dict], embeddings: list[list[float]]) -> int:
        """청크와 임베딩을 PostgreSQL에 저장합니다.

        Args:
            chunks: 저장할 청크 리스트. 각 항목은 'text'와 'metadata' 키를 포함.
            embeddings: 각 청크에 대응하는 임베딩 벡터 리스트.

        Returns:
            저장된 청크 수.

        Raises:
            ValueError: 임베딩 차원이 설정값과 다른 경우.
        """
        if not chunks:
            return 0

        # 차원 검증
        if embeddings and len(embeddings[0]) != self._dimension:
            actual_dim = len(embeddings[0])
            raise ValueError(
                f"임베딩 차원 불일치: 기대 {self._dimension}차원, 실제 {actual_dim}차원. "
                f"PGVECTOR_DIMENSION 설정을 확인하세요."
            )

        texts = [c["text"] for c in chunks]
        metadatas = [c.get("metadata", {}) for c in chunks]

        # langchain_postgres는 사전 계산된 임베딩을 add_embeddings으로 저장
        self._store.add_embeddings(
            texts=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        return len(chunks)

    def search(self, query_embedding: list[float], top_k: int = 10) -> list[dict]:
        """벡터 유사도 검색을 수행합니다.

        Args:
            query_embedding: 질의 임베딩 벡터.
            top_k: 반환할 최대 결과 수.

        Returns:
            유사도 순으로 정렬된 딕셔너리 리스트.
            각 항목: {text: str, metadata: dict, score: float}
            score = 1 - cosine_distance (0~1 범위).
        """
        results = self._store.similarity_search_with_score_by_vector(
            embedding=query_embedding,
            k=top_k,
        )

        output = []
        for doc, distance in results:
            output.append({
                "text": doc.page_content,
                "metadata": doc.metadata or {},
                "score": float(1 - distance),  # 코사인 거리를 유사도로 변환
            })
        return output

    def count(self) -> int:
        """저장된 청크의 총 수를 반환합니다.

        SQL로 직접 조회합니다.

        Returns:
            현재 저장된 청크 수.
        """
        try:
            with self._store._make_sync_session() as session:
                result = session.execute(
                    text(
                        "SELECT count(*) FROM langchain_pg_embedding "
                        "WHERE collection_id = ("
                        "  SELECT uuid FROM langchain_pg_collection WHERE name = :name"
                        ")"
                    ),
                    {"name": self._collection_name},
                )
                row = result.fetchone()
                return int(row[0]) if row else 0
        except Exception:
            return 0

    def get_all_texts(self) -> list[str]:
        """BM25 검색을 위해 저장된 모든 텍스트를 반환합니다.

        Returns:
            저장된 모든 청크 텍스트 리스트.
        """
        try:
            with self._store._make_sync_session() as session:
                result = session.execute(
                    text(
                        "SELECT document FROM langchain_pg_embedding "
                        "WHERE collection_id = ("
                        "  SELECT uuid FROM langchain_pg_collection WHERE name = :name"
                        ")"
                    ),
                    {"name": self._collection_name},
                )
                return [row[0] for row in result.fetchall() if row[0] is not None]
        except Exception:
            return []

    def get_all_texts_and_tokens(self) -> tuple[list[str], list[list[str]]]:
        """모든 텍스트와 사전 토큰화된 데이터를 반환합니다.

        BM25 인덱스 빌드를 위해 tokenized_text 메타데이터를 파싱합니다.
        tokenized_text는 공백 구분 문자열로 저장됩니다 (ChromaVectorStore 호환 형식).

        Returns:
            (원본 텍스트 리스트, 토큰 리스트의 리스트) 튜플.
        """
        try:
            with self._store._make_sync_session() as session:
                result = session.execute(
                    text(
                        "SELECT document, cmetadata->>'tokenized_text' "
                        "FROM langchain_pg_embedding "
                        "WHERE collection_id = ("
                        "  SELECT uuid FROM langchain_pg_collection WHERE name = :name"
                        ")"
                    ),
                    {"name": self._collection_name},
                )
                rows = result.fetchall()

            texts = []
            tokens_list = []
            for doc, token_str in rows:
                texts.append(doc or "")
                # ChromaVectorStore와 동일: 공백 구분 문자열을 리스트로 파싱
                tokens_list.append(token_str.split() if token_str else [])

            return texts, tokens_list
        except Exception:
            return [], []

    def list_collections(self) -> list[str]:
        """사용 가능한 컬렉션(테이블) 목록을 반환합니다.

        Returns:
            컬렉션 이름 문자열 리스트.
        """
        try:
            with self._store._make_sync_session() as session:
                result = session.execute(
                    text("SELECT name FROM langchain_pg_collection")
                )
                return [row[0] for row in result.fetchall()]
        except Exception:
            return []

    def get_all_documents(self) -> list[dict]:
        """저장된 모든 문서를 소스 파일 단위로 그룹화하여 반환합니다.

        Returns:
            소스 파일별 문서 리스트.
            각 항목: {source: str, chunk_count: int}
        """
        try:
            with self._store._make_sync_session() as session:
                result = session.execute(
                    text(
                        "SELECT cmetadata->>'source' as source, count(*) as cnt "
                        "FROM langchain_pg_embedding "
                        "WHERE collection_id = ("
                        "  SELECT uuid FROM langchain_pg_collection WHERE name = :name"
                        ") "
                        "GROUP BY cmetadata->>'source'"
                    ),
                    {"name": self._collection_name},
                )
                return [
                    {"source": row[0] or "unknown", "chunk_count": int(row[1])}
                    for row in result.fetchall()
                ]
        except Exception:
            return []

    def get_by_source(self, source: str) -> list[dict]:
        """특정 소스 문서의 모든 청크를 반환합니다.

        Args:
            source: 조회할 소스 파일명.

        Returns:
            해당 소스의 청크 리스트.
            각 항목: {chunk_id: str, text: str, metadata: dict}
        """
        try:
            with self._store._make_sync_session() as session:
                result = session.execute(
                    text(
                        "SELECT id, document, cmetadata "
                        "FROM langchain_pg_embedding "
                        "WHERE collection_id = ("
                        "  SELECT uuid FROM langchain_pg_collection WHERE name = :name"
                        ") "
                        "AND cmetadata->>'source' = :source"
                    ),
                    {"name": self._collection_name, "source": source},
                )
                rows = result.fetchall()

            return [
                {
                    "chunk_id": str(row[0]),
                    "text": row[1] or "",
                    "metadata": row[2] if isinstance(row[2], dict) else {},
                }
                for row in rows
            ]
        except Exception:
            return []

    def delete_by_source(self, source: str) -> int:
        """특정 소스 문서의 모든 청크를 삭제합니다.

        Args:
            source: 삭제할 소스 파일명.

        Returns:
            삭제된 청크 수. 존재하지 않으면 0.
        """
        try:
            with self._store._make_sync_session() as session:
                result = session.execute(
                    text(
                        "DELETE FROM langchain_pg_embedding "
                        "WHERE collection_id = ("
                        "  SELECT uuid FROM langchain_pg_collection WHERE name = :name"
                        ") "
                        "AND cmetadata->>'source' = :source"
                    ),
                    {"name": self._collection_name, "source": source},
                )
                session.commit()
                return result.rowcount
        except Exception:
            return 0

    def delete_all(self) -> int:
        """저장된 모든 데이터를 삭제합니다.

        컬렉션의 모든 임베딩을 삭제하고 동일한 이름으로 재생성합니다.

        Returns:
            삭제된 청크 수.
        """
        deleted_count = self.count()
        try:
            self._store.delete_collection()
        except Exception:
            pass

        # 컬렉션 재생성 (동일한 이름으로)
        try:
            self._store.create_collection()
        except Exception:
            pass

        return deleted_count


class _NoOpEmbeddings:
    """임베딩 계산 없이 사전 계산된 임베딩을 사용하기 위한 더미 임베딩 클래스.

    PGVector는 Embeddings 객체를 필요로 하지만, 우리는 임베딩을 직접 제공하므로
    실제 연산이 필요 없는 더미 구현을 사용합니다.
    """

    def __init__(self, dimension: int) -> None:
        """초기화

        Args:
            dimension: 임베딩 벡터 차원 수.
        """
        self._dimension = dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """문서 목록을 임베딩합니다 (더미 구현)."""
        return [[0.0] * self._dimension for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        """질의 텍스트를 임베딩합니다 (더미 구현)."""
        return [0.0] * self._dimension

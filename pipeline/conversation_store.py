# 대화 영속성 관리 모듈 - SQLite 기반 대화 및 메시지 CRUD
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path


class ConversationStore:
    """SQLite를 사용하여 대화 이력을 영속적으로 관리하는 클래스"""

    def __init__(self, db_path: str = "./storage/conversations.db"):
        """초기화

        Args:
            db_path: SQLite 데이터베이스 파일 경로 (기본값: ./storage/conversations.db)
        """
        # 부모 디렉토리가 없을 경우 생성
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # check_same_thread=False: FastAPI 비동기 환경 지원
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

        # WAL 모드: 동시 읽기 성능 향상
        self._conn.execute("PRAGMA journal_mode=WAL")
        # 외래키 제약 활성화: CASCADE 삭제 지원
        self._conn.execute("PRAGMA foreign_keys=ON")

        self._create_tables()

    def _create_tables(self) -> None:
        """데이터베이스 테이블 및 인덱스를 생성합니다."""
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_messages_conversation_id
                ON messages(conversation_id);
        """)
        self._conn.commit()

    def _now_iso(self) -> str:
        """현재 UTC 시각을 ISO 8601 형식 문자열로 반환합니다."""
        return datetime.now(timezone.utc).isoformat()

    def _row_to_conversation(self, row: sqlite3.Row) -> dict:
        """sqlite3.Row를 대화 딕셔너리로 변환합니다."""
        return {
            "id": row["id"],
            "title": row["title"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def create_conversation(self, title: str = "") -> dict:
        """새 대화를 생성합니다.

        Args:
            title: 대화 제목 (빈 문자열이면 '새 대화'로 설정)

        Returns:
            생성된 대화 딕셔너리 {"id", "title", "created_at", "updated_at"}
        """
        conversation_id = str(uuid.uuid4())
        now = self._now_iso()
        # 제목이 비어 있으면 기본 제목 사용
        resolved_title = title.strip() if title.strip() else "새 대화"

        self._conn.execute(
            "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (conversation_id, resolved_title, now, now),
        )
        self._conn.commit()

        return {
            "id": conversation_id,
            "title": resolved_title,
            "created_at": now,
            "updated_at": now,
        }

    def list_conversations(self) -> list[dict]:
        """모든 대화를 최신 순으로 반환합니다.

        Returns:
            대화 딕셔너리 리스트 (updated_at 내림차순 정렬)
        """
        cursor = self._conn.execute(
            "SELECT id, title, created_at, updated_at FROM conversations ORDER BY updated_at DESC"
        )
        return [self._row_to_conversation(row) for row in cursor.fetchall()]

    def get_conversation(self, conversation_id: str) -> dict | None:
        """특정 대화와 해당 메시지를 반환합니다.

        Args:
            conversation_id: 조회할 대화 ID

        Returns:
            메시지 포함 대화 딕셔너리, 없으면 None
            {"id", "title", "created_at", "updated_at", "messages": [{"role", "content", "created_at"}, ...]}
        """
        # 대화 조회
        cursor = self._conn.execute(
            "SELECT id, title, created_at, updated_at FROM conversations WHERE id = ?",
            (conversation_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None

        conversation = self._row_to_conversation(row)

        # 메시지 조회 (시간 오름차순)
        msg_cursor = self._conn.execute(
            "SELECT role, content, created_at FROM messages WHERE conversation_id = ? ORDER BY created_at ASC",
            (conversation_id,),
        )
        conversation["messages"] = [
            {
                "role": msg["role"],
                "content": msg["content"],
                "created_at": msg["created_at"],
            }
            for msg in msg_cursor.fetchall()
        ]

        return conversation

    def delete_conversation(self, conversation_id: str) -> bool:
        """대화와 연관 메시지를 삭제합니다.

        Args:
            conversation_id: 삭제할 대화 ID

        Returns:
            삭제 성공 시 True, 대화가 없으면 False
        """
        cursor = self._conn.execute(
            "DELETE FROM conversations WHERE id = ?",
            (conversation_id,),
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def update_title(self, conversation_id: str, title: str) -> dict | None:
        """대화 제목을 수정합니다.

        Args:
            conversation_id: 수정할 대화 ID
            title: 새 제목

        Returns:
            수정된 대화 딕셔너리, 대화가 없으면 None
        """
        now = self._now_iso()
        cursor = self._conn.execute(
            "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
            (title, now, conversation_id),
        )
        self._conn.commit()

        if cursor.rowcount == 0:
            return None

        return self.get_conversation(conversation_id)

    def add_messages(
        self,
        conversation_id: str,
        user_message: str,
        assistant_message: str,
    ) -> bool:
        """사용자/어시스턴트 메시지 쌍을 대화에 추가합니다.

        최초 메시지 추가 시 대화 제목이 '새 대화'이면 사용자 메시지
        앞 50자를 제목으로 자동 설정합니다.

        Args:
            conversation_id: 대상 대화 ID
            user_message: 사용자 메시지 내용
            assistant_message: 어시스턴트 메시지 내용

        Returns:
            추가 성공 시 True, 대화가 없으면 False
        """
        # 대화 존재 여부 및 현재 제목 확인
        cursor = self._conn.execute(
            "SELECT title FROM conversations WHERE id = ?",
            (conversation_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return False

        now = self._now_iso()

        try:
            # 사용자 메시지 삽입
            self._conn.execute(
                "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (conversation_id, "user", user_message, now),
            )
            # 어시스턴트 메시지 삽입
            self._conn.execute(
                "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (conversation_id, "assistant", assistant_message, now),
            )

            # 제목이 기본값이면 첫 번째 사용자 메시지로 자동 업데이트
            current_title = row["title"]
            if current_title == "새 대화":
                auto_title = user_message[:50].strip()
                self._conn.execute(
                    "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
                    (auto_title, now, conversation_id),
                )
            else:
                # updated_at 갱신
                self._conn.execute(
                    "UPDATE conversations SET updated_at = ? WHERE id = ?",
                    (now, conversation_id),
                )

            self._conn.commit()
            return True
        except sqlite3.Error:
            self._conn.rollback()
            return False

    def close(self) -> None:
        """데이터베이스 연결을 닫습니다."""
        self._conn.close()

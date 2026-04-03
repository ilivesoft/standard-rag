# 대화 저장소 테스트 - SQLite 기반 CRUD 검증
import pytest
from pipeline.conversation_store import ConversationStore


@pytest.fixture
def store(tmp_path):
    """격리된 ConversationStore 인스턴스"""
    db_path = str(tmp_path / "test_conversations.db")
    s = ConversationStore(db_path=db_path)
    yield s
    s.close()


class TestCreateConversation:
    """대화 생성 테스트"""

    def test_create_returns_dict(self, store):
        """생성 시 딕셔너리 반환"""
        result = store.create_conversation()
        assert isinstance(result, dict)
        assert "id" in result
        assert "title" in result
        assert "created_at" in result
        assert "updated_at" in result

    def test_create_with_title(self, store):
        """제목을 지정하여 생성"""
        result = store.create_conversation(title="테스트 대화")
        assert result["title"] == "테스트 대화"

    def test_create_without_title_sets_default(self, store):
        """제목 없이 생성 시 기본값 설정"""
        result = store.create_conversation()
        assert result["title"] == "새 대화"

    def test_create_generates_unique_ids(self, store):
        """각 대화에 고유 ID 생성"""
        c1 = store.create_conversation()
        c2 = store.create_conversation()
        assert c1["id"] != c2["id"]


class TestListConversations:
    """대화 목록 테스트"""

    def test_list_empty(self, store):
        """빈 상태에서 빈 리스트 반환"""
        result = store.list_conversations()
        assert result == []

    def test_list_returns_all(self, store):
        """모든 대화 반환"""
        store.create_conversation(title="대화 1")
        store.create_conversation(title="대화 2")
        result = store.list_conversations()
        assert len(result) == 2

    def test_list_ordered_by_updated_at_desc(self, store):
        """updated_at 기준 내림차순 정렬"""
        store.create_conversation(title="먼저 생성")
        store.create_conversation(title="나중 생성")
        result = store.list_conversations()
        # 나중에 생성된 것이 먼저 나와야 함
        assert result[0]["title"] == "나중 생성"


class TestGetConversation:
    """대화 상세 조회 테스트"""

    def test_get_existing(self, store):
        """존재하는 대화 조회"""
        created = store.create_conversation(title="테스트")
        result = store.get_conversation(created["id"])
        assert result is not None
        assert result["id"] == created["id"]
        assert result["title"] == "테스트"
        assert "messages" in result
        assert result["messages"] == []

    def test_get_nonexistent_returns_none(self, store):
        """존재하지 않는 대화 조회 시 None 반환"""
        result = store.get_conversation("nonexistent-id")
        assert result is None

    def test_get_with_messages(self, store):
        """메시지가 포함된 대화 조회"""
        created = store.create_conversation(title="테스트")
        store.add_messages(created["id"], "안녕하세요", "안녕하세요! 도움이 필요하신가요?")
        result = store.get_conversation(created["id"])
        assert len(result["messages"]) == 2
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][1]["role"] == "assistant"


class TestDeleteConversation:
    """대화 삭제 테스트"""

    def test_delete_existing(self, store):
        """존재하는 대화 삭제"""
        created = store.create_conversation()
        assert store.delete_conversation(created["id"]) is True
        assert store.get_conversation(created["id"]) is None

    def test_delete_nonexistent_returns_false(self, store):
        """존재하지 않는 대화 삭제 시 False 반환"""
        assert store.delete_conversation("nonexistent-id") is False

    def test_delete_cascades_messages(self, store):
        """대화 삭제 시 메시지도 함께 삭제"""
        created = store.create_conversation()
        store.add_messages(created["id"], "질문", "응답")
        store.delete_conversation(created["id"])
        assert store.get_conversation(created["id"]) is None


class TestUpdateTitle:
    """대화 제목 수정 테스트"""

    def test_update_title(self, store):
        """제목 수정 성공"""
        created = store.create_conversation(title="원래 제목")
        result = store.update_title(created["id"], "새 제목")
        assert result is not None
        assert result["title"] == "새 제목"

    def test_update_nonexistent_returns_none(self, store):
        """존재하지 않는 대화 수정 시 None 반환"""
        result = store.update_title("nonexistent-id", "새 제목")
        assert result is None


class TestAddMessages:
    """메시지 추가 테스트"""

    def test_add_messages(self, store):
        """메시지 쌍 추가"""
        created = store.create_conversation()
        result = store.add_messages(created["id"], "사용자 질문", "어시스턴트 응답")
        assert result is True

    def test_add_messages_to_nonexistent_returns_false(self, store):
        """존재하지 않는 대화에 메시지 추가 시 False 반환"""
        result = store.add_messages("nonexistent-id", "질문", "응답")
        assert result is False

    def test_add_messages_auto_title(self, store):
        """첫 메시지 추가 시 자동 제목 생성"""
        created = store.create_conversation()  # 기본 제목 "새 대화"
        store.add_messages(created["id"], "파이썬에서 리스트 컴프리헨션은 어떻게 사용하나요?", "응답")
        result = store.get_conversation(created["id"])
        assert result["title"] != "새 대화"
        assert len(result["title"]) <= 50

    def test_add_multiple_message_pairs(self, store):
        """여러 메시지 쌍 추가"""
        created = store.create_conversation(title="테스트")
        store.add_messages(created["id"], "질문1", "응답1")
        store.add_messages(created["id"], "질문2", "응답2")
        result = store.get_conversation(created["id"])
        assert len(result["messages"]) == 4

    def test_messages_ordered_chronologically(self, store):
        """메시지가 시간순으로 정렬"""
        created = store.create_conversation(title="테스트")
        store.add_messages(created["id"], "첫 질문", "첫 응답")
        store.add_messages(created["id"], "두 번째 질문", "두 번째 응답")
        result = store.get_conversation(created["id"])
        assert result["messages"][0]["content"] == "첫 질문"
        assert result["messages"][1]["content"] == "첫 응답"
        assert result["messages"][2]["content"] == "두 번째 질문"
        assert result["messages"][3]["content"] == "두 번째 응답"

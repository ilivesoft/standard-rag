# ConversationStore 인메모리 history 캐시 테스트
# 백엔드 누적 방식: get_cached_history / append_turn / invalidate_cache
import tempfile
from pathlib import Path

import pytest

from pipeline.conversation_store import ConversationStore


@pytest.fixture
def store():
    """임시 DB로 격리된 ConversationStore 인스턴스"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "test_conversations.db")
        s = ConversationStore(db_path=db_path)
        yield s
        s.close()


class TestGetCachedHistory:
    """get_cached_history 동작 검증"""

    def test_returns_empty_for_blank_conversation_id(self, store):
        """conversation_id가 빈 문자열이면 빈 리스트 반환"""
        assert store.get_cached_history("", max_turns=5) == []

    def test_returns_empty_for_unknown_conversation(self, store):
        """존재하지 않는 대화는 빈 리스트 반환 (DB rehydrate도 빈 결과)"""
        assert store.get_cached_history("non-existent-id", max_turns=5) == []

    def test_returns_appended_turns(self, store):
        """append_turn 후 동일 conversation_id로 조회하면 누적된 turn 반환"""
        conv = store.create_conversation(title="테스트")
        cid = conv["id"]

        store.append_turn(cid, "Q1", "A1")
        store.append_turn(cid, "Q2", "A2")

        history = store.get_cached_history(cid, max_turns=5)
        assert history == [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2"},
        ]

    def test_respects_max_turns_limit(self, store):
        """max_turns로 최근 N turn만 반환 (1 turn = 2 메시지)"""
        conv = store.create_conversation(title="테스트")
        cid = conv["id"]

        for i in range(4):
            store.append_turn(cid, f"Q{i}", f"A{i}")

        history = store.get_cached_history(cid, max_turns=2)
        assert len(history) == 4
        assert history[0]["content"] == "Q2"
        assert history[-1]["content"] == "A3"

    def test_max_turns_zero_returns_empty(self, store):
        """max_turns=0이면 빈 리스트 반환"""
        conv = store.create_conversation(title="테스트")
        cid = conv["id"]
        store.append_turn(cid, "Q", "A")

        assert store.get_cached_history(cid, max_turns=0) == []

    def test_returns_copy_not_reference(self, store):
        """반환된 리스트를 수정해도 내부 캐시는 영향받지 않음"""
        conv = store.create_conversation(title="테스트")
        cid = conv["id"]
        store.append_turn(cid, "Q1", "A1")

        h1 = store.get_cached_history(cid, max_turns=5)
        h1.append({"role": "user", "content": "변조"})

        h2 = store.get_cached_history(cid, max_turns=5)
        assert len(h2) == 2  # 변조가 반영되지 않음


class TestRehydrateFromDB:
    """캐시 미스 시 DB에서 1회 재수화(rehydrate)"""

    def test_rehydrates_existing_conversation_from_db(self, store):
        """기존 대화에 add_messages로 저장 후 캐시 없이 get_cached_history 호출 시 DB 로드"""
        conv = store.create_conversation(title="과거 대화")
        cid = conv["id"]
        # 캐시를 거치지 않고 직접 DB에만 저장
        store.add_messages(cid, user_message="이전질문", assistant_message="이전답변")

        # 이 시점에 캐시에는 없음 → DB rehydrate 발생
        history = store.get_cached_history(cid, max_turns=5)

        assert history == [
            {"role": "user", "content": "이전질문"},
            {"role": "assistant", "content": "이전답변"},
        ]

    def test_cache_is_populated_after_rehydrate(self, store):
        """rehydrate 이후에는 캐시에 존재하여 DB 재조회 없음"""
        conv = store.create_conversation(title="캐시 테스트")
        cid = conv["id"]
        store.add_messages(cid, user_message="Q", assistant_message="A")

        # 첫 호출: DB rehydrate
        store.get_cached_history(cid, max_turns=5)

        # 두 번째 호출 전 DB 연결을 끊어도 캐시에서 조회되어야 함
        store._conn.close()
        history = store.get_cached_history(cid, max_turns=5)

        assert len(history) == 2


class TestAppendTurn:
    """append_turn 동작 검증"""

    def test_append_turn_persists_to_db(self, store):
        """append_turn 후 DB에도 저장되어 재조회 가능"""
        conv = store.create_conversation(title="영속성")
        cid = conv["id"]

        store.append_turn(cid, "사용자메시지", "어시스턴트메시지")

        detail = store.get_conversation(cid)
        assert detail is not None
        assert len(detail["messages"]) == 2
        assert detail["messages"][0]["content"] == "사용자메시지"
        assert detail["messages"][1]["content"] == "어시스턴트메시지"

    def test_append_turn_returns_false_for_unknown_conversation(self, store):
        """존재하지 않는 대화 ID에 append_turn하면 False"""
        assert store.append_turn("non-existent", "Q", "A") is False

    def test_append_turn_returns_false_for_empty_id(self, store):
        """conversation_id가 빈 문자열이면 False"""
        assert store.append_turn("", "Q", "A") is False


class TestInvalidateCache:
    """invalidate_cache 동작 검증"""

    def test_invalidate_removes_cached_history(self, store):
        """invalidate_cache 호출 후 해당 대화의 캐시가 제거됨"""
        conv = store.create_conversation(title="무효화")
        cid = conv["id"]
        store.append_turn(cid, "Q", "A")

        # 캐시에 존재 확인
        assert len(store.get_cached_history(cid, max_turns=5)) == 2

        # 무효화 후 DB도 삭제
        store.invalidate_cache(cid)
        store.delete_conversation(cid)

        # 삭제 후 조회는 빈 리스트
        assert store.get_cached_history(cid, max_turns=5) == []

    def test_invalidate_unknown_id_is_noop(self, store):
        """존재하지 않는 ID를 invalidate해도 에러 없음"""
        store.invalidate_cache("non-existent-id")  # 예외 없이 동작

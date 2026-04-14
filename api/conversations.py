# 대화 관리 API 라우터 - 복수 대화 CRUD 엔드포인트
from fastapi import APIRouter, HTTPException

from models.conversation import (
    ConversationCreate,
    ConversationUpdate,
    ConversationResponse,
    ConversationDetailResponse,
    ConversationListResponse,
    AddMessagesRequest,
)

# 모듈 레벨 의존성 주입
_conversation_store = None


def set_conversation_store(store) -> None:
    """ConversationStore 인스턴스를 주입합니다."""
    global _conversation_store
    _conversation_store = store


def is_conversation_store_initialized() -> bool:
    """ConversationStore가 초기화되었는지 확인합니다."""
    return _conversation_store is not None


def get_conversation_store():
    """주입된 ConversationStore를 반환합니다. 미초기화 시 None."""
    return _conversation_store


def _get_store():
    """ConversationStore 인스턴스를 반환합니다. 미초기화 시 RuntimeError."""
    if _conversation_store is None:
        raise RuntimeError("ConversationStore가 초기화되지 않았습니다.")
    return _conversation_store


router = APIRouter(tags=["conversations"])


@router.post("/conversations", response_model=ConversationResponse)
async def create_conversation(request: ConversationCreate) -> ConversationResponse:
    """새 대화를 생성합니다."""
    store = _get_store()
    conversation = store.create_conversation(title=request.title)
    return ConversationResponse(**conversation)


@router.get("/conversations", response_model=ConversationListResponse)
async def list_conversations() -> ConversationListResponse:
    """대화 목록을 최신순으로 반환합니다."""
    store = _get_store()
    conversations = store.list_conversations()
    return ConversationListResponse(
        conversations=[ConversationResponse(**c) for c in conversations],
        total=len(conversations),
    )


@router.get("/conversations/{conversation_id}", response_model=ConversationDetailResponse)
async def get_conversation(conversation_id: str) -> ConversationDetailResponse:
    """특정 대화의 상세 정보와 메시지를 반환합니다."""
    store = _get_store()
    conversation = store.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="대화를 찾을 수 없습니다.")
    return ConversationDetailResponse(**conversation)


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str) -> dict:
    """대화와 관련된 모든 메시지를 삭제하고 인메모리 캐시도 무효화합니다."""
    store = _get_store()
    deleted = store.delete_conversation(conversation_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="대화를 찾을 수 없습니다.")
    store.invalidate_cache(conversation_id)
    return {"success": True}


@router.patch("/conversations/{conversation_id}", response_model=ConversationResponse)
async def update_conversation_title(
    conversation_id: str, request: ConversationUpdate
) -> ConversationResponse:
    """대화 제목을 수정합니다."""
    store = _get_store()
    conversation = store.update_title(conversation_id, request.title)
    if conversation is None:
        raise HTTPException(status_code=404, detail="대화를 찾을 수 없습니다.")
    return ConversationResponse(**conversation)


@router.post("/conversations/{conversation_id}/messages")
async def add_messages(
    conversation_id: str, request: AddMessagesRequest
) -> dict:
    """대화에 사용자/어시스턴트 메시지 쌍을 추가합니다."""
    store = _get_store()
    success = store.add_messages(
        conversation_id,
        user_message=request.user_message,
        assistant_message=request.assistant_message,
    )
    if not success:
        raise HTTPException(status_code=404, detail="대화를 찾을 수 없습니다.")
    return {"success": True}

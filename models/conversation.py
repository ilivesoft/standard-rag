# Pydantic v2 대화 스키마 - 대화 관리 API 입출력 모델
from pydantic import BaseModel, Field


class ConversationCreate(BaseModel):
    """대화 생성 요청 모델"""
    title: str = Field(default="", description="대화 제목 (비어있으면 자동 생성)")


class ConversationUpdate(BaseModel):
    """대화 제목 수정 요청 모델"""
    title: str = Field(description="새 대화 제목")


class ConversationResponse(BaseModel):
    """대화 응답 모델"""
    id: str
    title: str
    created_at: str
    updated_at: str


class MessageResponse(BaseModel):
    """메시지 응답 모델"""
    role: str
    content: str
    created_at: str


class ConversationDetailResponse(BaseModel):
    """대화 상세 응답 모델 (메시지 포함)"""
    id: str
    title: str
    created_at: str
    updated_at: str
    messages: list[MessageResponse]


class ConversationListResponse(BaseModel):
    """대화 목록 응답 모델"""
    conversations: list[ConversationResponse]
    total: int


class AddMessagesRequest(BaseModel):
    """메시지 쌍 추가 요청 모델"""
    user_message: str = Field(description="사용자 메시지")
    assistant_message: str = Field(description="어시스턴트 응답 메시지")

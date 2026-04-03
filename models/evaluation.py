# Pydantic v2 평가 스키마 - RAGAS 평가 입출력 모델
from typing import Optional
from pydantic import BaseModel, Field


class EvaluateRequest(BaseModel):
    """단일 평가 요청 모델"""
    question: str = Field(..., description="평가할 질문")
    answer: str = Field(..., description="LLM이 생성한 답변")
    contexts: list[str] = Field(..., description="검색된 컨텍스트 리스트")
    ground_truth: Optional[str] = Field(default=None, description="정답 (선택사항)")


class EvaluateResponse(BaseModel):
    """단일 평가 응답 모델"""
    faithfulness: float = Field(..., ge=0.0, le=1.0, description="충실성 점수")
    answer_relevancy: float = Field(..., ge=0.0, le=1.0, description="답변 관련성 점수")
    context_precision: float = Field(..., ge=0.0, le=1.0, description="컨텍스트 정밀도 점수")
    context_recall: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="컨텍스트 재현율 점수")


class BatchEvaluateRequest(BaseModel):
    """배치 평가 요청 모델"""
    items: list[EvaluateRequest] = Field(..., description="평가 항목 리스트")


class BatchEvaluateResponse(BaseModel):
    """배치 평가 응답 모델"""
    average: EvaluateResponse = Field(..., description="평균 점수")
    results: list[EvaluateResponse] = Field(..., description="개별 평가 결과")

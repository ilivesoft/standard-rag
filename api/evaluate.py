# 평가 API - /evaluate, /evaluate/batch 엔드포인트
from fastapi import APIRouter, HTTPException
from models.evaluation import (
    EvaluateRequest,
    EvaluateResponse,
    BatchEvaluateRequest,
    BatchEvaluateResponse,
)

# 의존성은 main.py에서 주입
_evaluator = None


def set_evaluator(evaluator):
    global _evaluator
    _evaluator = evaluator


def get_evaluator():
    return _evaluator


router = APIRouter(prefix="/evaluate", tags=["evaluate"])


@router.post("", response_model=EvaluateResponse)
async def evaluate(request: EvaluateRequest):
    """단일 질의-응답 쌍을 RAGAS로 평가합니다.

    Args:
        request: EvaluateRequest (question, answer, contexts, ground_truth)

    Returns:
        EvaluateResponse: 평가 점수
    """
    evaluator = get_evaluator()
    if evaluator is None:
        raise HTTPException(status_code=500, detail="평가기가 초기화되지 않았습니다.")

    try:
        result = evaluator.evaluate(
            question=request.question,
            answer=request.answer,
            contexts=request.contexts,
            ground_truth=request.ground_truth,
        )
        return EvaluateResponse(
            faithfulness=result.get("faithfulness", 0.0),
            answer_relevancy=result.get("answer_relevancy", 0.0),
            context_precision=result.get("context_precision", 0.0),
            context_recall=result.get("context_recall"),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch", response_model=BatchEvaluateResponse)
async def batch_evaluate(request: BatchEvaluateRequest):
    """여러 질의-응답 쌍을 배치로 평가합니다.

    Args:
        request: BatchEvaluateRequest (items 리스트)

    Returns:
        BatchEvaluateResponse: 평균 및 개별 평가 점수
    """
    evaluator = get_evaluator()
    if evaluator is None:
        raise HTTPException(status_code=500, detail="평가기가 초기화되지 않았습니다.")

    try:
        items = [
            {
                "question": item.question,
                "answer": item.answer,
                "contexts": item.contexts,
                "ground_truth": item.ground_truth,
            }
            for item in request.items
        ]
        result = evaluator.batch_evaluate(items)

        avg = result["average"]
        individual_results = result["results"]

        return BatchEvaluateResponse(
            average=EvaluateResponse(
                faithfulness=avg.get("faithfulness", 0.0),
                answer_relevancy=avg.get("answer_relevancy", 0.0),
                context_precision=avg.get("context_precision", 0.0),
                context_recall=avg.get("context_recall"),
            ),
            results=[
                EvaluateResponse(
                    faithfulness=r.get("faithfulness", 0.0),
                    answer_relevancy=r.get("answer_relevancy", 0.0),
                    context_precision=r.get("context_precision", 0.0),
                    context_recall=r.get("context_recall"),
                )
                for r in individual_results
            ],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

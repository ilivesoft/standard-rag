# RAG 평가 모듈 - RAGAS 기반 답변 품질 평가
from typing import Optional
from datasets import Dataset
from ragas import evaluate
try:
    # ragas >= 0.2.x 권장 임포트 경로
    from ragas.metrics.collections import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
except ImportError:
    # 이전 버전 호환성
    from ragas.metrics import (  # type: ignore
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )


class RAGEvaluator:
    """RAGAS를 사용하여 RAG 파이프라인의 품질을 평가하는 클래스"""

    # 평가에 사용할 기본 메트릭 목록
    _default_metrics = [faithfulness, answer_relevancy, context_precision]

    def evaluate(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: Optional[str] = None,
    ) -> dict:
        """단일 질의-응답 쌍을 평가합니다.

        Args:
            question: 평가할 질문
            answer: LLM이 생성한 답변
            contexts: 검색된 컨텍스트 리스트
            ground_truth: 정답 (context_recall 평가 시 필요)

        Returns:
            메트릭별 점수 딕셔너리
        """
        data = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
        }

        metrics = list(self._default_metrics)
        if ground_truth:
            data["ground_truth"] = [ground_truth]
            metrics.append(context_recall)

        dataset = Dataset.from_dict(data)
        result = evaluate(dataset, metrics=metrics)

        output = {
            "faithfulness": result["faithfulness"][0],
            "answer_relevancy": result["answer_relevancy"][0],
            "context_precision": result["context_precision"][0],
        }
        if ground_truth:
            output["context_recall"] = result["context_recall"][0]

        return output

    def batch_evaluate(self, items: list[dict]) -> dict:
        """여러 질의-응답 쌍을 배치로 평가하고 평균 메트릭을 계산합니다.

        Args:
            items: 평가 항목 리스트. 각 항목:
                   {"question": str, "answer": str, "contexts": list[str],
                    "ground_truth": Optional[str]}

        Returns:
            {"average": dict, "results": list[dict]}
        """
        questions = [it["question"] for it in items]
        answers = [it["answer"] for it in items]
        contexts_list = [it["contexts"] for it in items]
        ground_truths = [it.get("ground_truth") for it in items]

        has_ground_truth = any(gt for gt in ground_truths)

        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts_list,
        }

        metrics = list(self._default_metrics)
        if has_ground_truth:
            data["ground_truth"] = [gt or "" for gt in ground_truths]
            metrics.append(context_recall)

        dataset = Dataset.from_dict(data)
        result = evaluate(dataset, metrics=metrics)

        # 개별 결과 구성
        results = []
        for i in range(len(items)):
            item_result = {
                "faithfulness": result["faithfulness"][i],
                "answer_relevancy": result["answer_relevancy"][i],
                "context_precision": result["context_precision"][i],
            }
            if has_ground_truth:
                item_result["context_recall"] = result["context_recall"][i]
            results.append(item_result)

        # 평균 계산
        n = len(results)
        avg = {
            "faithfulness": sum(r["faithfulness"] for r in results) / n,
            "answer_relevancy": sum(r["answer_relevancy"] for r in results) / n,
            "context_precision": sum(r["context_precision"] for r in results) / n,
        }
        if has_ground_truth:
            avg["context_recall"] = sum(r.get("context_recall", 0) for r in results) / n

        return {"average": avg, "results": results}

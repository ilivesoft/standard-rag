# RAG 평가 모듈 테스트 - RAGAS 모킹, 배치 평가, 점수 범위 검증
import pytest
from unittest.mock import MagicMock, patch
from pipeline.evaluator import RAGEvaluator


@pytest.fixture
def mock_ragas_result():
    """RAGAS evaluate 반환값 모킹"""
    result = MagicMock()
    result.__getitem__ = lambda self, key: {
        "faithfulness": 0.85,
        "answer_relevancy": 0.90,
        "context_precision": 0.75,
        "context_recall": 0.80,
    }[key]
    return result


class TestRAGEvaluatorEvaluate:
    """evaluate 메서드 테스트"""

    def test_evaluate_returns_dict(self):
        """evaluate가 딕셔너리를 반환"""
        with patch("pipeline.evaluator.evaluate") as mock_eval:
            mock_result = MagicMock()
            mock_result.__getitem__ = lambda self, key: {
                "faithfulness": [0.85],
                "answer_relevancy": [0.90],
                "context_precision": [0.75],
            }[key]
            mock_eval.return_value = mock_result

            evaluator = RAGEvaluator()
            result = evaluator.evaluate(
                question="파이썬이란?",
                answer="파이썬은 범용 언어입니다.",
                contexts=["파이썬은 인터프리터 언어입니다."],
            )
            assert isinstance(result, dict)

    def test_evaluate_has_faithfulness(self):
        """결과에 faithfulness 키가 존재"""
        with patch("pipeline.evaluator.evaluate") as mock_eval:
            mock_result = MagicMock()
            mock_result.__getitem__ = lambda self, key: {
                "faithfulness": [0.85],
                "answer_relevancy": [0.90],
                "context_precision": [0.75],
            }[key]
            mock_eval.return_value = mock_result

            evaluator = RAGEvaluator()
            result = evaluator.evaluate(
                question="질문",
                answer="답변",
                contexts=["컨텍스트"],
            )
            assert "faithfulness" in result

    def test_evaluate_has_answer_relevancy(self):
        """결과에 answer_relevancy 키가 존재"""
        with patch("pipeline.evaluator.evaluate") as mock_eval:
            mock_result = MagicMock()
            mock_result.__getitem__ = lambda self, key: {
                "faithfulness": [0.85],
                "answer_relevancy": [0.90],
                "context_precision": [0.75],
            }[key]
            mock_eval.return_value = mock_result

            evaluator = RAGEvaluator()
            result = evaluator.evaluate(
                question="질문",
                answer="답변",
                contexts=["컨텍스트"],
            )
            assert "answer_relevancy" in result

    def test_evaluate_score_range(self):
        """모든 점수가 0~1 범위"""
        with patch("pipeline.evaluator.evaluate") as mock_eval:
            mock_result = MagicMock()
            mock_result.__getitem__ = lambda self, key: {
                "faithfulness": [0.85],
                "answer_relevancy": [0.90],
                "context_precision": [0.75],
            }[key]
            mock_eval.return_value = mock_result

            evaluator = RAGEvaluator()
            result = evaluator.evaluate(
                question="질문",
                answer="답변",
                contexts=["컨텍스트"],
            )
            for key in ["faithfulness", "answer_relevancy", "context_precision"]:
                if key in result and result[key] is not None:
                    assert 0.0 <= result[key] <= 1.0


class TestRAGEvaluatorBatchEvaluate:
    """batch_evaluate 메서드 테스트"""

    def test_batch_evaluate_returns_dict(self):
        """batch_evaluate가 딕셔너리를 반환"""
        with patch("pipeline.evaluator.evaluate") as mock_eval:
            mock_result = MagicMock()
            mock_result.__getitem__ = lambda self, key: {
                "faithfulness": [0.85, 0.75],
                "answer_relevancy": [0.90, 0.80],
                "context_precision": [0.75, 0.65],
            }[key]
            mock_eval.return_value = mock_result

            evaluator = RAGEvaluator()
            items = [
                {"question": "질문1", "answer": "답변1", "contexts": ["컨텍스트1"]},
                {"question": "질문2", "answer": "답변2", "contexts": ["컨텍스트2"]},
            ]
            result = evaluator.batch_evaluate(items)
            assert isinstance(result, dict)

    def test_batch_evaluate_has_average(self):
        """batch_evaluate 결과에 average 키가 존재"""
        with patch("pipeline.evaluator.evaluate") as mock_eval:
            mock_result = MagicMock()
            mock_result.__getitem__ = lambda self, key: {
                "faithfulness": [0.85, 0.75],
                "answer_relevancy": [0.90, 0.80],
                "context_precision": [0.75, 0.65],
            }[key]
            mock_eval.return_value = mock_result

            evaluator = RAGEvaluator()
            items = [
                {"question": "질문1", "answer": "답변1", "contexts": ["컨텍스트1"]},
            ]
            result = evaluator.batch_evaluate(items)
            assert "average" in result

    def test_batch_evaluate_has_results(self):
        """batch_evaluate 결과에 results 키가 존재"""
        with patch("pipeline.evaluator.evaluate") as mock_eval:
            mock_result = MagicMock()
            mock_result.__getitem__ = lambda self, key: {
                "faithfulness": [0.85],
                "answer_relevancy": [0.90],
                "context_precision": [0.75],
            }[key]
            mock_eval.return_value = mock_result

            evaluator = RAGEvaluator()
            items = [
                {"question": "질문1", "answer": "답변1", "contexts": ["컨텍스트1"]},
            ]
            result = evaluator.batch_evaluate(items)
            assert "results" in result

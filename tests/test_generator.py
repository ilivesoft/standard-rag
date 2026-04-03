# LLM 생성 모듈 테스트 - LangChain LLM 모킹, 프로바이더 전환 검증
import pytest
from unittest.mock import MagicMock, patch
from pipeline.generator import ResponseGenerator


@pytest.fixture
def sample_chunks():
    """생성기 테스트용 샘플 청크"""
    return [
        {"text": "파이썬은 범용 프로그래밍 언어입니다.", "metadata": {"source": "doc1.txt"}},
        {"text": "파이썬은 읽기 쉽고 배우기 쉽습니다.", "metadata": {"source": "doc1.txt"}},
    ]


class TestResponseGeneratorGenerate:
    """generate 메서드 테스트"""

    def test_generate_returns_dict(self, sample_chunks):
        """generate가 딕셔너리를 반환"""
        with patch("pipeline.generator.ChatOllama") as mock_cls:
            mock_llm = MagicMock()
            mock_response = MagicMock()
            mock_response.content = "파이썬은 범용 언어입니다."
            mock_llm.invoke.return_value = mock_response
            mock_cls.return_value = mock_llm

            generator = ResponseGenerator(provider="ollama")
            result = generator.generate("파이썬이란?", sample_chunks)
            assert isinstance(result, dict)

    def test_generate_has_answer_key(self, sample_chunks):
        """결과에 'answer' 키가 존재"""
        with patch("pipeline.generator.ChatOllama") as mock_cls:
            mock_llm = MagicMock()
            mock_response = MagicMock()
            mock_response.content = "파이썬은 범용 언어입니다."
            mock_llm.invoke.return_value = mock_response
            mock_cls.return_value = mock_llm

            generator = ResponseGenerator(provider="ollama")
            result = generator.generate("파이썬이란?", sample_chunks)
            assert "answer" in result

    def test_generate_has_sources_key(self, sample_chunks):
        """결과에 'sources' 키가 존재"""
        with patch("pipeline.generator.ChatOllama") as mock_cls:
            mock_llm = MagicMock()
            mock_response = MagicMock()
            mock_response.content = "파이썬은 범용 언어입니다."
            mock_llm.invoke.return_value = mock_response
            mock_cls.return_value = mock_llm

            generator = ResponseGenerator(provider="ollama")
            result = generator.generate("파이썬이란?", sample_chunks)
            assert "sources" in result

    def test_generate_answer_is_string(self, sample_chunks):
        """answer가 문자열"""
        with patch("pipeline.generator.ChatOllama") as mock_cls:
            mock_llm = MagicMock()
            mock_response = MagicMock()
            mock_response.content = "테스트 응답입니다."
            mock_llm.invoke.return_value = mock_response
            mock_cls.return_value = mock_llm

            generator = ResponseGenerator(provider="ollama")
            result = generator.generate("테스트 질문", sample_chunks)
            assert isinstance(result["answer"], str)

    def test_generate_sources_is_list(self, sample_chunks):
        """sources가 리스트"""
        with patch("pipeline.generator.ChatOllama") as mock_cls:
            mock_llm = MagicMock()
            mock_response = MagicMock()
            mock_response.content = "응답"
            mock_llm.invoke.return_value = mock_response
            mock_cls.return_value = mock_llm

            generator = ResponseGenerator(provider="ollama")
            result = generator.generate("질문", sample_chunks)
            assert isinstance(result["sources"], list)


class TestResponseGeneratorProvider:
    """LLM 프로바이더 전환 테스트"""

    def test_ollama_provider_uses_chatollama(self):
        """ollama 프로바이더 설정 시 ChatOllama 사용"""
        with patch("pipeline.generator.ChatOllama") as mock_ollama:
            mock_llm = MagicMock()
            mock_response = MagicMock()
            mock_response.content = "응답"
            mock_llm.invoke.return_value = mock_response
            mock_ollama.return_value = mock_llm

            generator = ResponseGenerator(provider="ollama")
            generator.generate("질문", [{"text": "컨텍스트", "metadata": {}}])
            mock_ollama.assert_called_once()

    def test_openai_provider_uses_chatopenai(self):
        """openai 프로바이더 설정 시 ChatOpenAI 사용"""
        with patch("pipeline.generator.ChatOpenAI") as mock_openai:
            mock_llm = MagicMock()
            mock_response = MagicMock()
            mock_response.content = "응답"
            mock_llm.invoke.return_value = mock_response
            mock_openai.return_value = mock_llm

            generator = ResponseGenerator(provider="openai")
            generator.generate("질문", [{"text": "컨텍스트", "metadata": {}}])
            mock_openai.assert_called_once()


class TestResponseGeneratorBuildPrompt:
    """_build_prompt 메서드 테스트"""

    def test_build_prompt_contains_query(self, sample_chunks):
        """프롬프트에 질의가 포함"""
        with patch("pipeline.generator.ChatOllama"):
            generator = ResponseGenerator(provider="ollama")
            prompt = generator._build_prompt("파이썬이란?", "컨텍스트 내용")
            assert "파이썬이란?" in prompt

    def test_build_prompt_contains_context(self, sample_chunks):
        """프롬프트에 컨텍스트가 포함"""
        with patch("pipeline.generator.ChatOllama"):
            generator = ResponseGenerator(provider="ollama")
            prompt = generator._build_prompt("질문", "컨텍스트 내용입니다")
            assert "컨텍스트 내용입니다" in prompt

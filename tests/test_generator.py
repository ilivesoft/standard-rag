# LLM 생성 모듈 테스트 - LangChain LLM 모킹, 프로바이더 전환 검증
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
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


class TestResponseGeneratorStream:
    """generate_stream 메서드 테스트"""

    @pytest.mark.asyncio
    async def test_generate_stream_yields_tokens(self, sample_chunks):
        """generate_stream이 토큰을 yield"""
        async def mock_astream(messages):
            for token in ["파이썬은", " 범용", " 언어입니다."]:
                chunk = MagicMock()
                chunk.content = token
                yield chunk

        with patch("pipeline.generator.ChatOllama") as mock_cls:
            mock_llm = MagicMock()
            mock_llm.astream = mock_astream
            mock_cls.return_value = mock_llm

            generator = ResponseGenerator(provider="ollama")
            tokens = []
            async for token in generator.generate_stream("파이썬이란?", sample_chunks):
                tokens.append(token)

            assert len(tokens) > 0
            assert all(isinstance(t, str) for t in tokens)

    @pytest.mark.asyncio
    async def test_generate_stream_skips_empty_tokens(self, sample_chunks):
        """빈 content 토큰은 yield하지 않음"""
        async def mock_astream(messages):
            for token in ["텍스트", "", "더 많은 텍스트"]:
                chunk = MagicMock()
                chunk.content = token
                yield chunk

        with patch("pipeline.generator.ChatOllama") as mock_cls:
            mock_llm = MagicMock()
            mock_llm.astream = mock_astream
            mock_cls.return_value = mock_llm

            generator = ResponseGenerator(provider="ollama")
            tokens = []
            async for token in generator.generate_stream("질문", sample_chunks):
                tokens.append(token)

            assert "" not in tokens


class TestResponseGeneratorBuildMessages:
    """_build_messages 메서드 테스트 (대화 맥락 포함)"""

    def test_build_messages_without_history(self):
        """history가 없으면 SystemMessage + 현재 HumanMessage만 포함"""
        with patch("pipeline.generator.ChatOllama"):
            generator = ResponseGenerator(provider="ollama")
            messages = generator._build_messages("파이썬이란?", "컨텍스트 내용", None)

            assert len(messages) == 2
            assert isinstance(messages[0], SystemMessage)
            assert "컨텍스트 내용" in messages[0].content
            assert isinstance(messages[1], HumanMessage)
            assert messages[1].content == "파이썬이란?"

    def test_build_messages_with_history_order(self):
        """history가 있으면 System → pairs → 현재 Human 순서로 구성"""
        with patch("pipeline.generator.ChatOllama"):
            generator = ResponseGenerator(provider="ollama")
            history = [
                {"role": "user", "content": "A"},
                {"role": "assistant", "content": "B"},
                {"role": "user", "content": "C"},
                {"role": "assistant", "content": "D"},
            ]
            messages = generator._build_messages("지금 질문", "ctx", history)

            assert len(messages) == 6
            assert isinstance(messages[0], SystemMessage)
            assert isinstance(messages[1], HumanMessage) and messages[1].content == "A"
            assert isinstance(messages[2], AIMessage) and messages[2].content == "B"
            assert isinstance(messages[3], HumanMessage) and messages[3].content == "C"
            assert isinstance(messages[4], AIMessage) and messages[4].content == "D"
            assert isinstance(messages[5], HumanMessage) and messages[5].content == "지금 질문"

    def test_build_messages_skips_empty_and_unknown_roles(self):
        """빈 content나 알 수 없는 role은 건너뜀"""
        with patch("pipeline.generator.ChatOllama"):
            generator = ResponseGenerator(provider="ollama")
            history = [
                {"role": "user", "content": ""},
                {"role": "system", "content": "무시됨"},
                {"role": "assistant", "content": "유효"},
            ]
            messages = generator._build_messages("q", "ctx", history)

            # System + AIMessage("유효") + HumanMessage("q")
            assert len(messages) == 3
            assert isinstance(messages[1], AIMessage)
            assert messages[1].content == "유효"

    def test_generate_passes_history_to_llm(self, sample_chunks):
        """generate가 history를 LLM invoke에 전달"""
        with patch("pipeline.generator.ChatOllama") as mock_cls:
            mock_llm = MagicMock()
            mock_response = MagicMock()
            mock_response.content = "응답"
            mock_llm.invoke.return_value = mock_response
            mock_cls.return_value = mock_llm

            generator = ResponseGenerator(provider="ollama")
            history = [
                {"role": "user", "content": "이전 질문"},
                {"role": "assistant", "content": "이전 답변"},
            ]
            generator.generate("현재 질문", sample_chunks, history=history)

            # invoke에 전달된 메시지 리스트 확인
            called_messages = mock_llm.invoke.call_args[0][0]
            contents = [m.content for m in called_messages]
            assert any("이전 질문" in c for c in contents)
            assert any("이전 답변" in c for c in contents)
            assert "현재 질문" in contents[-1]

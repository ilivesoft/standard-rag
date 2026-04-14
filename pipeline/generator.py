# LLM 응답 생성 모듈 - Ollama/OpenAI 지원, LangChain 기반
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


class ResponseGenerator:
    """LLM을 사용하여 컨텍스트 기반 응답을 생성하는 클래스"""

    def __init__(self, provider: str = "ollama", timeout: int = 30):
        """초기화

        Args:
            provider: 사용할 LLM 제공자 ('ollama' 또는 'openai')
            timeout: LLM 호출 타임아웃 (초)
        """
        self._provider = provider
        self._timeout = timeout

    def _prepare_messages(
        self,
        query: str,
        context_chunks: list[dict],
        history: list[dict] | None = None,
    ) -> list:
        """컨텍스트 청크에서 메시지 리스트를 구성합니다.

        Args:
            query: 사용자 질의
            context_chunks: 컨텍스트로 사용할 청크 리스트
            history: 이전 대화 메시지 리스트 (user/assistant role, 시간 오름차순)

        Returns:
            LangChain BaseMessage 인스턴스 리스트
        """
        context = "\n\n".join([c["text"] for c in context_chunks])
        return self._build_messages(query, context, history)

    async def generate_stream(
        self,
        query: str,
        context_chunks: list[dict],
        history: list[dict] | None = None,
    ):
        """스트리밍 방식으로 LLM 응답을 생성합니다.

        Args:
            query: 사용자 질의
            context_chunks: 컨텍스트로 사용할 청크 리스트
            history: 이전 대화 메시지 리스트 (user/assistant role, 시간 오름차순)

        Yields:
            응답 텍스트 토큰 (str)
        """
        messages = self._prepare_messages(query, context_chunks, history)
        llm = self._get_llm()
        async for chunk in llm.astream(messages):
            if chunk.content:
                yield chunk.content

    def generate(
        self,
        query: str,
        context_chunks: list[dict],
        history: list[dict] | None = None,
    ) -> dict:
        """컨텍스트 기반으로 LLM 응답을 생성합니다.

        Args:
            query: 사용자 질의
            context_chunks: 컨텍스트로 사용할 청크 리스트
            history: 이전 대화 메시지 리스트 (user/assistant role, 시간 오름차순)

        Returns:
            {"answer": str, "sources": list[dict]}
        """
        messages = self._prepare_messages(query, context_chunks, history)
        llm = self._get_llm()
        response = llm.invoke(messages)

        # 소스 메타데이터 추출
        sources = [c.get("metadata", {}) for c in context_chunks]

        return {
            "answer": response.content,
            "sources": sources,
        }

    def _build_messages(
        self,
        query: str,
        context: str,
        history: list[dict] | None = None,
    ) -> list:
        """LLM에 전달할 LangChain 메시지 리스트를 구성합니다.

        구조:
            SystemMessage(컨텍스트 포함)
            [과거 turn들: HumanMessage / AIMessage 쌍]
            HumanMessage(현재 질의)

        Args:
            query: 현재 사용자 질의
            context: 검색된 컨텍스트 텍스트
            history: 이전 대화 메시지 (role: "user"|"assistant", content: str)

        Returns:
            LangChain BaseMessage 인스턴스 리스트
        """
        system_prompt = (
            "다음 컨텍스트를 바탕으로 대화를 이어가며 질문에 답하세요. "
            "이전 대화의 맥락을 고려해 답하되, 사실 근거는 컨텍스트에서 찾으세요.\n\n"
            f"컨텍스트:\n{context}"
        )
        messages: list = [SystemMessage(content=system_prompt)]

        for msg in history or []:
            role = msg.get("role")
            content = msg.get("content", "")
            if not content:
                continue
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
            # 다른 role은 무시

        messages.append(HumanMessage(content=query))
        return messages

    def _get_llm(self):
        """LLM_PROVIDER에 따라 적절한 LLM 인스턴스를 반환합니다."""
        from config.settings import settings

        if self._provider == "openai":
            return ChatOpenAI(
                model=settings.OPENAI_MODEL,
                api_key=settings.OPENAI_API_KEY,
                timeout=self._timeout,
            )
        else:
            # 기본값: Ollama
            return ChatOllama(
                model=settings.OLLAMA_MODEL,
                base_url=settings.OLLAMA_BASE_URL,
                timeout=self._timeout,
            )

    @property
    def is_connected(self) -> bool:
        """LLM 서비스 연결 상태를 확인합니다."""
        try:
            llm = self._get_llm()
            # 간단한 테스트 호출
            llm.invoke([HumanMessage(content="ping")])
            return True
        except Exception:
            return False

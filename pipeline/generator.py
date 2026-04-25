# LLM 응답 생성 모듈 - Ollama/OpenAI 지원, LangChain LCEL 기반
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


class ResponseGenerator:
    """LLM을 사용하여 컨텍스트 기반 응답을 생성하는 클래스"""

    _PROMPT = ChatPromptTemplate.from_messages([
        ("system", (
            "다음 컨텍스트를 바탕으로 대화를 이어가며 질문에 답하세요. "
            "이전 대화의 맥락을 고려해 답하되, 사실 근거는 컨텍스트에서 찾으세요.\n\n"
            "컨텍스트:\n{context}"
        )),
        MessagesPlaceholder("history"),
        ("human", "{query}"),
    ])

    def __init__(self, provider: str = "ollama", timeout: int = 30):
        """초기화

        Args:
            provider: 사용할 LLM 제공자 ('ollama' 또는 'openai')
            timeout: LLM 호출 타임아웃 (초)
        """
        self._provider = provider
        self._timeout = timeout

    def _build_history(self, history: list[dict] | None) -> list:
        """대화 이력 딕셔너리를 LangChain 메시지 객체 리스트로 변환합니다."""
        messages = []
        for msg in history or []:
            content = msg.get("content", "")
            if not content:
                continue
            if msg.get("role") == "user":
                messages.append(HumanMessage(content=content))
            elif msg.get("role") == "assistant":
                messages.append(AIMessage(content=content))
        return messages

    def _build_chain(self):
        """LCEL 체인을 구성합니다."""
        return self._PROMPT | self._get_llm()

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
        context = "\n\n".join(c["text"] for c in context_chunks)
        chain = self._build_chain() | StrOutputParser()
        async for chunk in chain.astream({
            "context": context,
            "history": self._build_history(history),
            "query": query,
        }):
            yield chunk

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
        context = "\n\n".join(c["text"] for c in context_chunks)
        response = self._build_chain().invoke({
            "context": context,
            "history": self._build_history(history),
            "query": query,
        })
        return {
            "answer": response.content,
            "sources": [c.get("metadata", {}) for c in context_chunks],
        }

    def _get_llm(self):
        """LLM_PROVIDER에 따라 적절한 LLM 인스턴스를 반환합니다."""
        from config.settings import settings

        if self._provider == "openai":
            return ChatOpenAI(
                model=settings.OPENAI_MODEL,
                api_key=settings.OPENAI_API_KEY,
                timeout=self._timeout,
            )
        # 기본값: Ollama
        return ChatOllama(
            model=settings.OLLAMA_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            timeout=self._timeout,
        )

    _REWRITE_PROMPT = ChatPromptTemplate.from_messages([
        ("system", (
            "주어진 검색 쿼리를 문서 검색에 최적화된 더 구체적인 형태로 재작성하세요. "
            "재작성된 쿼리만 출력하고, 다른 설명은 포함하지 마세요."
        )),
        ("human", "{query}"),
    ])

    def rewrite_query(self, query: str) -> str:
        """검색 품질이 낮을 때 쿼리를 재작성합니다.

        Args:
            query: 원본 검색 쿼리

        Returns:
            재작성된 쿼리 문자열 (실패 시 원본 쿼리 반환)
        """
        try:
            chain = self._REWRITE_PROMPT | self._get_llm() | StrOutputParser()
            return chain.invoke({"query": query}).strip()
        except Exception:
            return query

    @property
    def is_connected(self) -> bool:
        """LLM 서비스 연결 상태를 확인합니다."""
        try:
            self._get_llm().invoke([HumanMessage(content="ping")])
            return True
        except Exception:
            return False

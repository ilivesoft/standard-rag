# LLM 응답 생성 모듈 - Ollama/OpenAI 지원, LangChain 기반
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


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

    def generate(self, query: str, context_chunks: list[dict]) -> dict:
        """컨텍스트 기반으로 LLM 응답을 생성합니다.

        Args:
            query: 사용자 질의
            context_chunks: 컨텍스트로 사용할 청크 리스트

        Returns:
            {"answer": str, "sources": list[dict]}
        """
        # 컨텍스트 텍스트 조합
        context = "\n\n".join([c["text"] for c in context_chunks])
        prompt = self._build_prompt(query, context)

        llm = self._get_llm()
        response = llm.invoke([HumanMessage(content=prompt)])

        # 소스 메타데이터 추출
        sources = [c.get("metadata", {}) for c in context_chunks]

        return {
            "answer": response.content,
            "sources": sources,
        }

    def _build_prompt(self, query: str, context: str) -> str:
        """LLM에 전달할 프롬프트를 구성합니다.

        Args:
            query: 사용자 질의
            context: 검색된 컨텍스트 텍스트

        Returns:
            완성된 프롬프트 문자열
        """
        return f"""다음 컨텍스트를 바탕으로 질문에 답하세요.

컨텍스트:
{context}

질문: {query}

답변:"""

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

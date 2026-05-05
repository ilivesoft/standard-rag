# LangGraph 질의 워크플로우 - retrieve→rerank→generate, 빈 컬렉션 조건부 라우팅
from __future__ import annotations

from langgraph.graph import StateGraph, END
from pipeline.graphs.state import QueryState
from pipeline.vectorstore_protocol import VectorStoreProtocol


def create_query_graph(retriever, reranker, generator, vectorstore: VectorStoreProtocol, settings=None):
    """질의 응답 파이프라인 LangGraph를 생성하여 반환합니다.

    Args:
        retriever: HybridRetriever 인스턴스
        reranker: DocumentReranker 인스턴스
        generator: ResponseGenerator 인스턴스
        vectorstore: VectorStoreProtocol 구현체 (컬렉션 크기 확인용)
        settings: Settings 인스턴스 (품질 임계값/최대 재시도 수 설정용, None이면 기본값 사용)

    Returns:
        컴파일된 LangGraph 워크플로우
    """
    quality_threshold: float = getattr(settings, "QUALITY_THRESHOLD", 0.0)
    max_retries: int = getattr(settings, "MAX_RETRIES", 2)

    def _node_error_guard(state: QueryState):
        """이전 노드에서 오류가 발생한 경우 현재 상태를 그대로 반환합니다.
        오류가 없으면 None을 반환하여 노드 실행을 계속합니다."""
        if state.get("error"):
            return state
        return None

    def retrieve_node(state: QueryState) -> QueryState:
        """하이브리드 검색을 수행합니다."""
        # 빈 컬렉션 확인
        if vectorstore.count() == 0:
            return {**state, "is_empty_collection": True, "retrieved_chunks": []}

        try:
            chunks = retriever.retrieve(
                state["query"],
                top_k=state.get("top_k", 10),
                alpha=state.get("alpha", 0.5),
            )
            return {**state, "retrieved_chunks": chunks, "is_empty_collection": False}
        except Exception as e:
            return {**state, "error": str(e)}

    def rerank_node(state: QueryState) -> QueryState:
        """검색 결과를 재순위화합니다."""
        if (guard := _node_error_guard(state)) is not None:
            return guard
        try:
            reranked = reranker.rerank(
                state["query"],
                state["retrieved_chunks"],
                top_n=state.get("top_n", 3),
            )
            return {**state, "reranked_chunks": reranked}
        except Exception as e:
            return {**state, "error": str(e)}

    def grade_node(state: QueryState) -> QueryState:
        """reranked_chunks의 rerank_score로 검색 품질을 평가합니다."""
        chunks = state.get("reranked_chunks", [])
        if not chunks:
            return {**state, "quality_score": -999.0}
        scores = [c.get("rerank_score", -999.0) for c in chunks]
        return {**state, "quality_score": max(scores)}
    
    def generate_node(state: QueryState) -> QueryState:
        """재순위된 청크를 기반으로 응답을 생성합니다."""
        if (guard := _node_error_guard(state)) is not None:
            return guard
        try:
            result = generator.generate(
                state["query"],
                state["reranked_chunks"],
                history=state.get("chat_history", []),
            )
            return {
                **state,
                "answer": result["answer"],
                "sources": result["sources"],
            }
        except Exception as e:
            return {**state, "error": str(e)}

    def no_docs_node(state: QueryState) -> QueryState:
        """관련 문서가 없을 때 안내 메시지를 반환합니다."""
        if state.get("is_empty_collection"):
            msg = "인덱싱된 문서가 없습니다. 먼저 문서를 업로드하여 인덱싱해 주세요."
        else:
            msg = "관련 문서를 찾을 수 없습니다. 다른 키워드로 검색해 보세요."
        return {**state, "answer": msg, "sources": []}

    def query_rewrite_node(state: QueryState) -> QueryState:
        """검색 품질이 낮을 때 LLM으로 쿼리를 재작성하고 재검색을 준비합니다."""
        if (guard := _node_error_guard(state)) is not None:
            return guard
        try:
            rewritten = generator.rewrite_query(state["query"])
            return {
                **state,
                "query": rewritten,
                "retrieved_chunks": [],
                "reranked_chunks": [],
                "quality_score": 0.0,
                "retry_count": state.get("retry_count", 0) + 1,
            }
        except Exception as e:
            return {**state, "error": str(e)}

    def route_after_retrieve(state: QueryState) -> str:
        """retrieve 후 라우팅 결정: 빈 컬렉션이거나 품질 게이트 실패(빈 결과)이면 no_docs, 아니면 rerank"""
        if state.get("is_empty_collection") or not state.get("retrieved_chunks"):
            return "no_docs"
        return "rerank"

    def route_after_grade(state: QueryState) -> str:
        """grade 후 라우팅 결정: 품질 충분하거나 최대 재시도 도달 시 generate, 아니면 query_rewrite"""
        if state.get("quality_score", -999.0) >= quality_threshold:
            return "generate"
        if state.get("retry_count", 0) >= max_retries:
            return "generate"
        return "query_rewrite"

    # 그래프 구성
    graph = StateGraph(QueryState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("grade", grade_node)
    graph.add_node("generate", generate_node)
    graph.add_node("no_docs", no_docs_node)
    graph.add_node("query_rewrite", query_rewrite_node)

    graph.set_entry_point("retrieve")
    graph.add_conditional_edges(
        "retrieve",
        route_after_retrieve,
        {"rerank": "rerank", "no_docs": "no_docs"},
    )
    graph.add_edge("rerank", "grade")
    graph.add_conditional_edges(
        "grade",
        route_after_grade,
        {"generate": "generate", "query_rewrite": "query_rewrite"},
    )
    graph.add_edge("query_rewrite", "retrieve")
    graph.add_edge("generate", END)
    graph.add_edge("no_docs", END)

    compiled = graph.compile()

    # Mermaid 코드로 그래프 시각화하여 파일로 저장
    mermaid_code = compiled.get_graph().draw_mermaid()
    with open("pipeline/graphs/query_graph.mmd", "w", encoding="utf-8") as f:
        f.write(mermaid_code)

    return compiled

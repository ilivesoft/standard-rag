# LangGraph 질의 워크플로우 - retrieve→rerank→generate, 빈 컬렉션 조건부 라우팅
from langgraph.graph import StateGraph, END
from pipeline.graphs.state import QueryState


def create_query_graph(retriever, reranker, generator, vectorstore):
    """질의 응답 파이프라인 LangGraph를 생성하여 반환합니다.

    Args:
        retriever: HybridRetriever 인스턴스
        reranker: DocumentReranker 인스턴스
        generator: ResponseGenerator 인스턴스
        vectorstore: VectorStore 인스턴스 (컬렉션 크기 확인용)

    Returns:
        컴파일된 LangGraph 워크플로우
    """

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
        if state.get("error"):
            return state
        try:
            reranked = reranker.rerank(
                state["query"],
                state["retrieved_chunks"],
                top_n=state.get("top_n", 3),
            )
            return {**state, "reranked_chunks": reranked}
        except Exception as e:
            return {**state, "error": str(e)}

    def generate_node(state: QueryState) -> QueryState:
        """재순위된 청크를 기반으로 응답을 생성합니다."""
        if state.get("error"):
            return state
        try:
            result = generator.generate(state["query"], state["reranked_chunks"])
            return {
                **state,
                "answer": result["answer"],
                "sources": result["sources"],
            }
        except Exception as e:
            return {**state, "error": str(e)}

    def no_docs_node(state: QueryState) -> QueryState:
        """인덱싱된 문서가 없을 때 안내 메시지를 반환합니다."""
        return {
            **state,
            "answer": "인덱싱된 문서가 없습니다. 먼저 문서를 업로드하여 인덱싱해 주세요.",
            "sources": [],
        }

    def route_after_retrieve(state: QueryState) -> str:
        """retrieve 후 라우팅 결정: 빈 컬렉션이면 no_docs, 아니면 rerank"""
        if state.get("is_empty_collection"):
            return "no_docs"
        return "rerank"

    # 그래프 구성
    graph = StateGraph(QueryState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("generate", generate_node)
    graph.add_node("no_docs", no_docs_node)

    graph.set_entry_point("retrieve")
    graph.add_conditional_edges(
        "retrieve",
        route_after_retrieve,
        {"rerank": "rerank", "no_docs": "no_docs"},
    )
    graph.add_edge("rerank", "generate")
    graph.add_edge("generate", END)
    graph.add_edge("no_docs", END)

    return graph.compile()

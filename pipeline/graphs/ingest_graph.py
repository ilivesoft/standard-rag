# LangGraph 인덱싱 워크플로우 - parse→clean→chunk→tokenize→embed→store 순서 실행
from pathlib import Path
from langgraph.graph import StateGraph, END
from pipeline.graphs.state import IngestState
from pipeline.tokenizer import tokenize_korean


def create_ingest_graph(parser, cleaner, chunker, embedder, vectorstore):
    """인덱싱 파이프라인 LangGraph를 생성하여 반환합니다.

    Args:
        parser: DocumentParser 인스턴스
        cleaner: TextCleaner 인스턴스
        chunker: TextChunker 인스턴스
        embedder: DocumentEmbedder 인스턴스
        vectorstore: VectorStore 인스턴스

    Returns:
        컴파일된 LangGraph 워크플로우
    """

    def parse_node(state: IngestState) -> IngestState:
        """파일을 파싱하여 원시 텍스트 추출"""
        try:
            raw_text = parser.parse(Path(state["file_path"]))
            return {**state, "raw_text": raw_text}
        except Exception as e:
            return {**state, "error": str(e)}

    def clean_node(state: IngestState) -> IngestState:
        """원시 텍스트를 정제"""
        if state.get("error"):
            return state
        try:
            cleaned = cleaner.clean(state["raw_text"])
            return {**state, "cleaned_text": cleaned}
        except Exception as e:
            return {**state, "error": str(e)}

    def chunk_node(state: IngestState) -> IngestState:
        """정제된 텍스트를 청크로 분할"""
        if state.get("error"):
            return state
        try:
            file_path = Path(state["file_path"])
            metadata = {"source": file_path.name}
            chunks = chunker.chunk(state["cleaned_text"], metadata)
            return {**state, "chunks": chunks}
        except Exception as e:
            return {**state, "error": str(e)}

    def tokenize_node(state: IngestState) -> IngestState:
        """청크 텍스트를 BM25 검색용으로 토큰화하여 메타데이터에 저장"""
        if state.get("error"):
            return state
        try:
            tokenized_chunks = []
            for chunk in state["chunks"]:
                tokens = tokenize_korean(chunk["text"])
                metadata = {**chunk["metadata"], "tokenized_text": " ".join(tokens)}
                tokenized_chunks.append({**chunk, "metadata": metadata})
            return {**state, "chunks": tokenized_chunks}
        except Exception as e:
            return {**state, "error": str(e)}

    def embed_node(state: IngestState) -> IngestState:
        """청크를 임베딩 벡터로 변환"""
        if state.get("error"):
            return state
        try:
            texts = [c["text"] for c in state["chunks"]]
            embeddings = embedder.embed(texts)
            return {**state, "embeddings": embeddings}
        except Exception as e:
            return {**state, "error": str(e)}

    def store_node(state: IngestState) -> IngestState:
        """임베딩을 벡터 저장소에 저장"""
        if state.get("error"):
            return state
        try:
            count = vectorstore.add(state["chunks"], state["embeddings"])
            return {**state, "stored_count": count}
        except Exception as e:
            return {**state, "error": str(e)}

    # 그래프 구성
    graph = StateGraph(IngestState)
    graph.add_node("parse", parse_node)
    graph.add_node("clean", clean_node)
    graph.add_node("chunk", chunk_node)
    graph.add_node("tokenize", tokenize_node)
    graph.add_node("embed", embed_node)
    graph.add_node("store", store_node)

    graph.set_entry_point("parse")
    graph.add_edge("parse", "clean")
    graph.add_edge("clean", "chunk")
    graph.add_edge("chunk", "tokenize")
    graph.add_edge("tokenize", "embed")
    graph.add_edge("embed", "store")
    graph.add_edge("store", END)
    
    compiled = graph.compile()

    # Mermaid 코드로 그래프 시각화하여 파일로 저장
    mermaid_code = compiled.get_graph().draw_mermaid()
    with open("pipeline/graphs/ingest_graph.mmd", "w", encoding="utf-8") as f:
        f.write(mermaid_code)

    return compiled

# Gradio 데모 UI 진입점 - RAG 관리, 챗봇 인터페이스
import json
import logging
import os
import httpx
import gradio as gr

from config.settings import settings

logger = logging.getLogger(__name__)

# HTTP 타임아웃 상수
_HTTP_LONG_TIMEOUT = 300   # 파일 업로드 / LLM 질의용
_HTTP_SHORT_TIMEOUT = 10   # 목록 조회 / 삭제 등 빠른 요청용

# UI 슬라이더 제한
_TOP_K_MAX = 50
_TOP_N_MAX = 10

# API 서버 기본 URL
API_BASE_URL = "http://localhost:8000"


def upload_files(files) -> str:
    """파일을 API 서버에 업로드하고 인덱싱합니다."""
    if not files:
        return "파일을 선택해주세요."

    results = []
    for file_path in files:
        try:
            with open(file_path, "rb") as f:
                filename = os.path.basename(file_path)
                response = httpx.post(
                    f"{API_BASE_URL}/ingest/file",
                    files={"file": (filename, f)},
                    timeout=_HTTP_LONG_TIMEOUT,
                )
            if response.status_code == 200:
                data = response.json()
                if data.get("skipped"):
                    results.append(f"건너뜀 - {data['filename']}: {data.get('skip_reason', '')}")
                else:
                    results.append(f"완료 - {data['filename']}: {data['chunks_indexed']}개 청크 인덱싱")
            else:
                results.append(f"오류 - {response.text}")
        except Exception as e:
            results.append(f"오류 - {str(e)}")

    return "\n".join(results)


def upload_and_return(files):
    """파일을 업로드한 후 목록 화면으로 전환합니다."""
    result = upload_files(files)
    rows, status = refresh_rag_documents()
    return result, rows, status, gr.update(visible=True), gr.update(visible=False)


def refresh_rag_documents():
    """RAG 인덱스 문서 목록을 조회합니다."""
    try:
        response = httpx.get(f"{API_BASE_URL}/index/documents", timeout=_HTTP_SHORT_TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            rows = [[False, doc["source"], doc["chunk_count"]] for doc in data.get("documents", [])]
            total = f"총 {data['total_documents']}개 문서 / {data['total_chunks']}개 청크"
            return rows, total
    except Exception as e:
        logger.warning("RAG 문서 목록 조회 실패: %s", e)
    return [], "서버에 연결할 수 없습니다"


def delete_selected_documents(table_data):
    """체크된 문서를 RAG 인덱스에서 삭제합니다."""
    if table_data is None:
        return refresh_rag_documents()

    rows = table_data.values.tolist() if hasattr(table_data, "values") else table_data

    for row in rows:
        if row and row[0]:
            source = str(row[1])
            try:
                httpx.delete(f"{API_BASE_URL}/index/documents/{source}", timeout=_HTTP_SHORT_TIMEOUT)
            except Exception as e:
                logger.warning("문서 삭제 실패 (%s): %s", source, e)

    return refresh_rag_documents()


def show_upload_panel():
    """문서 업로드 화면으로 전환합니다."""
    return gr.update(visible=False), gr.update(visible=True)


def show_list_panel():
    """RAG 문서 목록 화면으로 전환합니다."""
    rows, status = refresh_rag_documents()
    return gr.update(visible=True), gr.update(visible=False), rows, status


def _append_error_to_history(history: list, message: str, error_msg: str) -> None:
    """사용자 메시지와 에러 응답을 히스토리에 추가합니다."""
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": error_msg})


def chat(message: str, history: list, top_k: int, top_n: int, alpha: float, conversation_id: str):
    """질의를 API 서버에 전송하고 응답을 반환합니다.

    대화 맥락은 백엔드가 conversation_id 기반으로 누적 관리합니다. 프론트엔드는
    화면 표시용으로만 history를 유지하고, 질의 시 history를 전송하지 않습니다.
    신규 대화는 백엔드가 생성하여 응답의 conversation_id로 반환합니다.
    """
    if not message.strip():
        return "", history, None, conversation_id, False

    was_new = not conversation_id

    try:
        response = httpx.post(
            f"{API_BASE_URL}/query",
            json={
                "query": message,
                "top_k": top_k,
                "top_n": top_n,
                "alpha": alpha,
                "conversation_id": conversation_id,
            },
            timeout=_HTTP_LONG_TIMEOUT,
        )
        if response.status_code == 200:
            data = response.json()
            answer = data.get("answer", "응답 없음")
            sources = data.get("sources", [])
            # 백엔드가 신규 대화를 생성했으면 응답에서 ID를 받아 사용
            conversation_id = data.get("conversation_id", conversation_id)

            # 화면 표시용으로만 history를 누적 (LLM 맥락은 백엔드가 관리)
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": answer})

            new_conv_created = was_new and bool(conversation_id)
            return "", history, sources, conversation_id, new_conv_created
        else:
            error_msg = f"오류: {response.text}"
            _append_error_to_history(history, message, error_msg)
            return "", history, None, conversation_id, False
    except Exception as e:
        error_msg = f"연결 오류: {str(e)}"
        _append_error_to_history(history, message, error_msg)
        return "", history, None, conversation_id, False

def load_conversation_list() -> list:
    """대화 목록을 조회하여 반환합니다. [[title, "⋯", id], ...]"""
    try:
        response = httpx.get(f"{API_BASE_URL}/conversations", timeout=_HTTP_SHORT_TIMEOUT)
        if response.status_code == 200:
            conversations = response.json().get("conversations", [])
            return [[c["title"] or "제목 없음", "⋯", c["id"]] for c in conversations]
    except Exception as e:
        logger.warning("대화 목록 조회 실패: %s", e)
    return []


def maybe_reload_conv_list(new_conv_created: bool):
    """새 대화가 생성된 경우에만 대화 목록을 API에서 다시 조회합니다."""
    if new_conv_created:
        return load_conversation_list()
    return gr.update()


def new_conversation(current_sources):
    """채팅 화면을 초기화합니다. 대화는 첫 질문 전송 시 생성됩니다."""
    clear = None if current_sources else gr.update()
    return [], "", "", load_conversation_list(), "", gr.update(visible=False), clear


def on_conv_select(selected_data, current_sources, evt: gr.SelectData):
    """대화 항목 선택 시 대화를 불러오거나 ⋯ 클릭 시 액션 패널을 표시합니다."""
    try:
        if selected_data is not None:
            rows = selected_data.values.tolist() if hasattr(selected_data, "values") else selected_data
            if rows and evt.index[0] < len(rows):
                row = rows[evt.index[0]]
                conv_id = str(row[2])
                title = str(row[0])
                clear = None if current_sources else gr.update()

                col_idx = evt.index[1] if isinstance(evt.index, (list, tuple)) and len(evt.index) > 1 else 0
                if col_idx == 1:  # ⋯ 클릭 → 액션 패널 표시 (출처 유지)
                    return gr.update(), conv_id, gr.update(visible=True), title, gr.update()
                # 제목 클릭 → 대화 불러오기 (출처 조건부 초기화)
                response = httpx.get(f"{API_BASE_URL}/conversations/{conv_id}", timeout=_HTTP_SHORT_TIMEOUT)
                if response.status_code == 200:
                    data = response.json()
                    history = [
                        {"role": m["role"], "content": m["content"]}
                        for m in data.get("messages", [])
                    ]
                    return history, conv_id, gr.update(visible=False), title, clear
    except Exception as e:
        logger.warning("대화 선택 처리 실패: %s", e)
    return gr.update(), "", gr.update(visible=False), "", None if current_sources else gr.update()


def delete_conversation(conversation_id: str):
    """현재 채팅을 삭제합니다."""
    if not conversation_id:
        return [], "", load_conversation_list(), "", gr.update(visible=False)
    try:
        httpx.delete(f"{API_BASE_URL}/conversations/{conversation_id}", timeout=_HTTP_SHORT_TIMEOUT)
    except Exception as e:
        logger.warning("대화 삭제 실패 (%s): %s", conversation_id, e)
    return [], "", load_conversation_list(), "", gr.update(visible=False)


def rename_conversation(conv_id: str, new_title: str):
    """대화 제목을 변경합니다."""
    if conv_id and new_title.strip():
        try:
            httpx.patch(
                f"{API_BASE_URL}/conversations/{conv_id}",
                json={"title": new_title.strip()},
                timeout=_HTTP_SHORT_TIMEOUT,
            )
        except Exception as e:
            logger.warning("대화 제목 변경 실패 (%s): %s", conv_id, e)
    return load_conversation_list()


def load_initial_data():
    """앱 시작 시 RAG 문서 목록과 대화 목록을 초기 로드합니다."""
    rag_rows, rag_status = refresh_rag_documents()
    return rag_rows, rag_status, load_conversation_list()


def _lock_input(current_sources) -> tuple:
    """전송 중 입력창을 비활성화하고 출처 컨트롤이 있을 경우 초기화합니다."""
    clear = None if current_sources else gr.update()
    return gr.update(interactive=False, placeholder="처리 중..."), clear


def _unlock_input() -> gr.update:
    """전송 완료 후 입력창을 활성화하고 원래 placeholder를 복원합니다."""
    return gr.update(interactive=True, placeholder="질문을 입력하세요...")


def create_demo() -> gr.Blocks:
    """Gradio 데모 UI를 생성합니다."""
    css = """
    #conv-list thead { display: none !important; }
    #conv-list td:nth-child(3),
    #conv-list th:nth-child(3) { display: none !important; }
    #conv-list td:nth-child(2) { width: 30px !important; min-width: 30px !important; max-width: 30px !important; text-align: center !important; opacity: 0; cursor: pointer !important; color: #888; transition: opacity 0.15s; padding: 0 !important; }
    #conv-list tr:hover td:nth-child(2) { opacity: 1 !important; }
    #conv-list { --color-accent: var(--border-color-primary, #e5e7eb); --border-color-accent: var(--border-color-primary, #e5e7eb); }
    #conv-list td:focus, #conv-list td:focus-visible, #conv-list td:focus-within { outline: none !important; box-shadow: none !important; border-color: var(--border-color-primary, #e5e7eb) !important; }

    #rag-doc-table { --color-accent: var(--border-color-primary, #e5e7eb); --border-color-accent: var(--border-color-primary, #e5e7eb); }
    """

    with gr.Blocks(title="Standard RAG", css=css) as demo:
        gr.Markdown("## Standard RAG")
        gr.Markdown("문서를 업로드한 뒤 검색 인덱스를 관리하고, 질의응답 결과를 확인해보세요.")

        with gr.Tab("RAG 관리"):
            # --- 목록 화면 (기본 표시) ---
            with gr.Column(visible=True) as rag_list_panel:
                with gr.Row():
                    gr.HTML("", elem_classes=["flex-spacer"])
                    rag_upload_btn = gr.Button("문서 업로드", variant="huggingface")
                    rag_refresh_btn = gr.Button("목록 새로고침", variant="secondary")
                    rag_delete_selected_btn = gr.Button("선택 삭제", variant="stop")
                rag_index_status = gr.Textbox(label="RAG 인덱스 상태", interactive=False)
                rag_document_table = gr.Dataframe(
                    headers=["선택", "파일명", "청크 수"],
                    datatype=["bool", "str", "number"],
                    column_widths=["80px", "1fr", "120px"],
                    label="인덱싱된 문서 목록 (체크 후 선택 삭제)",
                    interactive=True,
                    elem_id="rag-doc-table",
                )

            # --- 문서 업로드 화면 (숨김, 전환 시 표시) ---
            with gr.Column(visible=False) as rag_upload_panel:
                with gr.Row():
                    gr.HTML("", elem_classes=["flex-spacer"])
                    rag_back_btn = gr.Button("← 목록 돌아가기", variant="secondary")
                    upload_btn = gr.Button("업로드 및 인덱싱", variant="huggingface")
                file_upload = gr.File(
                    file_count="multiple",
                    label="파일 선택 (PDF, DOCX, TXT, MD, HTML, PNG, JPG, JPEG, BMP, TIFF)",
                    file_types=[
                        ".pdf", ".docx", ".txt", ".md", ".html",
                        ".png", ".jpg", ".jpeg", ".bmp", ".tiff",
                    ],
                )
                upload_result = gr.Textbox(label="업로드 결과", lines=3, interactive=False)

            rag_upload_btn.click(fn=show_upload_panel, inputs=[], outputs=[rag_list_panel, rag_upload_panel])
            rag_refresh_btn.click(fn=refresh_rag_documents, inputs=[], outputs=[rag_document_table, rag_index_status])
            rag_delete_selected_btn.click(
                fn=delete_selected_documents,
                inputs=[rag_document_table],
                outputs=[rag_document_table, rag_index_status],
            )
            rag_back_btn.click(
                fn=show_list_panel,
                inputs=[],
                outputs=[rag_list_panel, rag_upload_panel, rag_document_table, rag_index_status],
            )
            upload_btn.click(
                fn=upload_and_return,
                inputs=[file_upload],
                outputs=[upload_result, rag_document_table, rag_index_status, rag_list_panel, rag_upload_panel],
            )

        with gr.Tab("챗봇"):
            conversation_id_state = gr.State(value="")
            new_conv_created_state = gr.State(value=False)

            with gr.Row():
                # 좌측 사이드바: 대화 목록
                with gr.Column(scale=1):
                    with gr.Row():
                        new_chat_btn = gr.Button("새 채팅", variant="secondary")
                    gr.Markdown("최근 항목")
                    with gr.Row():
                        conversation_list = gr.Dataframe(
                            headers=["제목", " ", "ID"],
                            interactive=False,
                            show_label=False,
                            elem_id="conv-list",
                        )
                    with gr.Row(visible=False) as action_panel:
                        with gr.Column(scale=1):
                            rename_input = gr.Textbox(
                                placeholder="새 제목 입력...",
                                show_label=False,
                                scale=3,
                            )
                            rename_btn = gr.Button("저장", variant="huggingface")
                            delete_chat_btn = gr.Button("삭제", variant="huggingface")

                # 우측 메인: 채팅 영역
                with gr.Column(scale=4):
                    chatbot = gr.Chatbot(label="대화", height=400, elem_id="chatbot")
                    with gr.Row():
                        query_input = gr.Textbox(
                            label="질문",
                            placeholder="질문을 입력하세요...",
                            scale=4,
                        )
                        submit_btn = gr.Button("전송", variant="primary", scale=1)

                    with gr.Row():
                        with gr.Column(scale=3):
                            top_k_slider = gr.Slider(minimum=1, maximum=_TOP_K_MAX, value=settings.TOP_K_RETRIEVAL, step=1, label="검색 결과 수 (top_k)")
                            top_n_slider = gr.Slider(minimum=1, maximum=_TOP_N_MAX, value=settings.TOP_N_RERANK, step=1, label="재순위 결과 수 (top_n)")
                            alpha_slider = gr.Slider(minimum=0.0, maximum=1.0, value=settings.HYBRID_ALPHA, step=0.1, label="하이브리드 알파 (0=BM25, 1=Vector)")
                        with gr.Column(scale=7):
                            sources = gr.JSON(label="출처", height=220)

            # 이벤트 핸들러
            new_chat_btn.click(
                fn=new_conversation,
                inputs=[sources],
                outputs=[chatbot, query_input, conversation_id_state, conversation_list, rename_input, action_panel, sources],
            )
            delete_chat_btn.click(
                fn=delete_conversation,
                inputs=[conversation_id_state],
                outputs=[chatbot, conversation_id_state, conversation_list, rename_input, action_panel],
            )
            conversation_list.select(
                fn=on_conv_select,
                inputs=[conversation_list, sources],
                outputs=[chatbot, conversation_id_state, action_panel, rename_input, sources],
            )
            rename_btn.click(
                fn=rename_conversation,
                inputs=[conversation_id_state, rename_input],
                outputs=[conversation_list],
            )
            submit_btn.click(
                fn=_lock_input, inputs=[sources], outputs=[query_input, sources], queue=False,
            ).then(
                fn=chat,
                inputs=[query_input, chatbot, top_k_slider, top_n_slider, alpha_slider, conversation_id_state],
                outputs=[query_input, chatbot, sources, conversation_id_state, new_conv_created_state],
            ).then(
                fn=_unlock_input, inputs=[], outputs=[query_input], queue=False,
            ).then(
                fn=maybe_reload_conv_list,
                inputs=[new_conv_created_state],
                outputs=[conversation_list],
            )
            query_input.submit(
                fn=_lock_input, inputs=[sources], outputs=[query_input, sources], queue=False,
            ).then(
                fn=chat,
                inputs=[query_input, chatbot, top_k_slider, top_n_slider, alpha_slider, conversation_id_state],
                outputs=[query_input, chatbot, sources, conversation_id_state, new_conv_created_state],
            ).then(
                fn=_unlock_input, inputs=[], outputs=[query_input], queue=False,
            ).then(
                fn=maybe_reload_conv_list,
                inputs=[new_conv_created_state],
                outputs=[conversation_list],
            )

        demo.load(
            fn=load_initial_data,
            inputs=[],
            outputs=[rag_document_table, rag_index_status, conversation_list],
        )

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(server_name="127.0.0.1", server_port=7860)

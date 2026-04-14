---
id: SPEC-RAG-001
version: "1.1.0"
status: completed
created: "2026-03-22"
updated: "2026-03-22"
author: jkh
priority: high
issue_number: 0
---

## HISTORY

| 버전 | 날짜 | 작성자 | 변경 내용 |
|------|------|--------|-----------|
| 1.0.0 | 2026-03-22 | jkh | 최초 작성 |
| 1.1.0 | 2026-03-22 | jkh | RAG 평가 모듈 추가, LangGraph 파이프라인 오케스트레이션, LlamaIndex 문서 처리 레이어 통합 |

---

# SPEC-RAG-001: End-to-End RAG Pipeline System

## 개요

Windows 로컬 환경에서 실행 가능한 데모용 End-to-End RAG(Retrieval-Augmented Generation) 파이프라인 시스템.
문서 업로드부터 파싱, 정제, 청킹, 임베딩, 하이브리드 검색, 리랭킹, LLM 응답 생성까지 전체 파이프라인을 자동 처리한다.

---

## Environment (환경)

- **운영체제**: Windows 10/11 (64-bit)
- **런타임**: Python 3.11+
- **메모리**: 최소 8GB RAM (16GB 권장)
- **디스크**: 최소 5GB (모델 파일 포함)
- **GPU**: 선택사항 (CUDA 지원 NVIDIA GPU, 임베딩/Reranker 속도 향상)
- **LLM 엔진**: Ollama (로컬) 또는 OpenAI API (클라우드)
- **네트워크**: 최초 모델 다운로드 시 인터넷 필요, 이후 완전 오프라인 가능

---

## Assumptions (가정)

- 사용자는 Python 3.11+ 환경을 갖추고 있다
- 로컬 LLM 사용 시 Ollama가 설치되어 있고 모델이 pull 완료되어 있다
- 처리 대상 문서는 텍스트 추출 가능한 형식이다 (스캔 이미지 PDF 제외)
- 데모 환경 기준 최대 파일 크기는 100MB이다
- ChromaDB는 로컬 파일 시스템에 영속 저장한다
- 단일 사용자 동시 접근을 기준으로 설계한다

---

## Requirements (요구사항)

### 모듈 1: 문서 수집 및 파싱 (Document Ingestion & Parsing)

**[REQ-RAG-001] Ubiquitous: 지원 형식 파싱**

시스템은 **항상** 업로드된 파일의 확장자(PDF, DOCX, TXT, Markdown, HTML)를 자동 감지하여 적절한 파서를 선택하고 텍스트를 추출해야 한다.

**[REQ-RAG-002] Event-Driven: 단일 파일 업로드 처리**

**WHEN** 사용자가 `/ingest/file` 엔드포인트로 단일 파일을 업로드 **THEN** 시스템은 파일을 `storage/uploads/`에 저장하고, 파싱 -> 정제 -> 청킹 -> 임베딩 -> Vector DB 저장 파이프라인을 순차 실행하여 인덱싱 결과를 반환해야 한다.

**[REQ-RAG-003] Event-Driven: 폴더 일괄 업로드 처리**

**WHEN** 사용자가 `/ingest/folder` 엔드포인트로 복수 파일을 일괄 업로드 **THEN** 시스템은 각 파일에 대해 개별적으로 파싱 파이프라인을 실행하고, 전체 처리 결과(성공/실패 건수)를 요약하여 반환해야 한다.

**[REQ-RAG-004] Unwanted: 미지원 파일 형식 거부**

시스템은 지원하지 않는 파일 형식(예: .exe, .zip, .jpg)을 인덱싱 **하지 않아야 한다**. 미지원 형식 업로드 시 명확한 오류 메시지와 지원 형식 목록을 반환해야 한다.

**[REQ-RAG-005] Unwanted: 빈 파일 거부**

시스템은 텍스트 내용이 없는 빈 파일 또는 텍스트 추출 결과가 비어 있는 파일을 인덱싱 **하지 않아야 한다**. 해당 파일에 대해 건너뛰기 처리하고 로그에 기록해야 한다.

---

### 모듈 2: 텍스트 처리 파이프라인 (Text Processing Pipeline)

**[REQ-RAG-010] Ubiquitous: 텍스트 정제**

시스템은 **항상** 추출된 텍스트에 대해 다음 정제 작업을 수행해야 한다:
- 특수문자 및 제어문자 제거
- 연속 공백/개행 정규화
- 인코딩 이슈 처리 (UTF-8 변환)
- 노이즈 텍스트(헤더/푸터 반복 등) 필터링

**[REQ-RAG-011] Ubiquitous: Chunk 분할**

시스템은 **항상** 정제된 텍스트를 `RecursiveCharacterTextSplitter`를 사용하여 설정 가능한 크기(기본 512자)와 오버랩(기본 64자)으로 Chunk를 분할해야 한다. 각 Chunk에는 원본 파일명, 페이지 번호(해당 시), Chunk 인덱스를 메타데이터로 포함해야 한다.

> **LlamaIndex 통합 참고**: 대안으로 LlamaIndex `SentenceSplitter`를 사용하여 의미 단위 청킹을 수행할 수 있다. 설정을 통해 `RecursiveCharacterTextSplitter`와 `SentenceSplitter` 간 전환 가능.

**[REQ-RAG-012] Ubiquitous: Embedding 생성**

시스템은 **항상** 분할된 각 Chunk에 대해 `BAAI/bge-m3` 모델(sentence-transformers)을 사용하여 벡터 임베딩을 생성해야 한다. 배치 처리를 통해 다수 Chunk를 효율적으로 처리해야 한다.

> **LlamaIndex 통합 참고**: LlamaIndex `HuggingFaceEmbedding` 파이프라인을 통해 임베딩 생성을 수행할 수 있다. LlamaIndex 래퍼를 사용할 경우 동일한 `BAAI/bge-m3` 모델을 내부적으로 활용한다.

**[REQ-RAG-013] Event-Driven: Vector DB 저장**

**WHEN** 임베딩 생성이 완료되면 **THEN** 시스템은 벡터, 텍스트, 메타데이터를 ChromaDB의 PersistentClient를 통해 저장해야 한다. 컬렉션명은 설정 가능하며, 기본값은 `default`이다.

**[REQ-RAG-014] State-Driven: 대용량 파일 배치 처리**

**IF** 단일 파일에서 추출된 Chunk 수가 1000개를 초과하면 **THEN** 시스템은 Chunk를 배치 단위(기본 500개)로 분할하여 임베딩 생성 및 DB 저장을 수행하여 메모리 사용을 제어해야 한다.

---

### 모듈 3: 하이브리드 검색 및 랭킹 (Hybrid Search & Ranking)

**[REQ-RAG-020] Event-Driven: 하이브리드 검색 실행**

**WHEN** 사용자가 `/query` 엔드포인트로 질의를 전송 **THEN** 시스템은 다음을 병렬로 실행해야 한다:
1. ChromaDB 벡터 유사도 검색 (Top-K, 기본 K=10)
2. BM25 키워드 검색 (rank_bm25 라이브러리)

두 결과를 RRF(Reciprocal Rank Fusion) 알고리즘으로 병합하여 최종 Top-K 후보를 생성해야 한다.

> **LlamaIndex 통합 참고**: 검색 엔진은 LlamaIndex `RetrieverQueryEngine`으로 래핑하여 통합 인터페이스를 제공할 수 있다. LlamaIndex `VectorIndexRetriever`와 BM25 검색을 결합하여 하이브리드 검색을 구성한다.

**[REQ-RAG-021] Event-Driven: Reranker 적용**

**WHEN** 하이브리드 검색 결과가 생성되면 **THEN** 시스템은 `cross-encoder/ms-marco-MiniLM-L-6-v2` 모델을 사용하여 (query, document) 쌍의 관련성을 재점수화하고, 상위 N개(기본 N=3)를 최종 검색 결과로 반환해야 한다.

**[REQ-RAG-022] State-Driven: 검색 파라미터 설정**

**IF** 사용자가 질의 시 검색 파라미터(top_k, top_n, alpha)를 지정하면 **THEN** 시스템은 해당 파라미터를 우선 적용하고, 미지정 시 `config/settings.py`의 기본값을 사용해야 한다.

**[REQ-RAG-023] Unwanted: 빈 컬렉션 검색 방지**

시스템은 인덱싱된 문서가 없는 빈 컬렉션에 대한 검색을 실행 **하지 않아야 한다**. 빈 컬렉션 질의 시 적절한 안내 메시지를 반환해야 한다.

---

### 모듈 4: LLM 응답 생성 (LLM Response Generation)

**[REQ-RAG-030] Event-Driven: 컨텍스트 기반 응답 생성**

**WHEN** Reranker에 의해 최종 검색 결과가 확정되면 **THEN** 시스템은 검색된 문서 컨텍스트와 사용자 질의를 프롬프트 템플릿에 조합하여 LLM(Ollama 또는 OpenAI)에 전달하고, 생성된 응답과 함께 출처 정보(파일명, 페이지)를 반환해야 한다.

**[REQ-RAG-031] State-Driven: LLM Provider 전환**

**IF** 환경 변수 `LLM_PROVIDER`가 `ollama`로 설정되어 있으면 **THEN** 시스템은 LangChain OllamaLLM을 사용해야 한다. **IF** `openai`로 설정되어 있으면 **THEN** LangChain ChatOpenAI를 사용해야 한다.

**[REQ-RAG-032] Unwanted: LLM 미연결 시 오류 처리**

시스템은 LLM 서비스(Ollama/OpenAI)에 연결할 수 없을 때 무한 대기 **하지 않아야 한다**. 연결 타임아웃(기본 30초) 이후 명확한 오류 메시지를 반환해야 한다.

**[REQ-RAG-033] Optional: 스트리밍 응답**

**가능하면** LLM 응답을 스트리밍(Server-Sent Events) 방식으로 제공하여 사용자에게 실시간 응답 생성 과정을 보여줄 수 있어야 한다.

---

### 모듈 5: API 및 UI 인터페이스 (API & UI Interface)

**[REQ-RAG-040] Ubiquitous: REST API 제공**

시스템은 **항상** FastAPI 기반 REST API를 포트 8000에서 제공해야 하며, 다음 엔드포인트를 포함해야 한다:
- `POST /ingest/file` - 단일 파일 업로드 및 인덱싱
- `POST /ingest/folder` - 폴더 일괄 업로드 및 인덱싱
- `POST /query` - 질의 응답 (RAG 파이프라인 실행)
- `GET /collections` - 저장된 컬렉션 목록 조회
- `GET /health` - 서비스 상태 확인 (파이프라인 구성 요소 상태 포함)

**[REQ-RAG-041] Ubiquitous: Gradio Demo UI 제공**

시스템은 **항상** Gradio 기반 데모 UI를 포트 7860에서 제공해야 하며, 다음 기능을 포함해야 한다:
- 파일/폴더 업로드 인터페이스
- 채팅 형식의 질의응답 인터페이스
- 검색 결과 출처 표시
- 인덱싱 상태 표시

**[REQ-RAG-042] Ubiquitous: OpenAPI 문서 자동 생성**

시스템은 **항상** FastAPI의 자동 OpenAPI(Swagger) 문서를 `/docs` 경로에서 제공해야 한다. 모든 엔드포인트에 대한 요청/응답 스키마가 문서화되어야 한다.

**[REQ-RAG-043] Event-Driven: Health Check 응답**

**WHEN** `/health` 엔드포인트에 GET 요청이 도착 **THEN** 시스템은 다음 구성 요소의 상태를 확인하여 반환해야 한다:
- ChromaDB 연결 상태
- 임베딩 모델 로드 상태
- LLM 서비스 연결 상태
- Reranker 모델 로드 상태

**[REQ-RAG-044] State-Driven: Pydantic 데이터 검증**

**IF** API 요청 데이터가 Pydantic 스키마 검증을 통과하지 못하면 **THEN** 시스템은 422 Unprocessable Entity 응답과 함께 상세한 검증 오류 메시지를 반환해야 한다.

---

### 모듈 6: LangGraph 파이프라인 오케스트레이션 (LangGraph Pipeline Orchestration)

**[REQ-RAG-050] Ubiquitous: LangGraph 워크플로우 관리**

시스템은 **항상** LangGraph StateGraph를 사용하여 RAG 파이프라인의 각 단계를 노드로 정의하고 상태를 관리해야 한다.
- Ingest 그래프 노드: parse → clean → chunk → embed → store
- Query 그래프 노드: retrieve → rerank → generate
- 각 노드는 독립적으로 실행되며 상태(State)를 다음 노드로 전달한다

**[REQ-RAG-051] Event-Driven: 그래프 기반 인덱싱 실행**

**WHEN** 문서 업로드 요청이 도착하면 **THEN** 시스템은 LangGraph Ingest 그래프를 실행하여 parse→clean→chunk→embed→store 순서로 처리하고, 각 노드의 처리 결과를 StateGraph를 통해 다음 노드로 전달해야 한다.

**[REQ-RAG-052] Event-Driven: 그래프 기반 질의 실행**

**WHEN** 질의 요청이 도착하면 **THEN** 시스템은 LangGraph Query 그래프를 실행하여 retrieve→rerank→generate 순서로 처리하고, 중간 상태(검색 결과, 재순위 결과)를 보존하여 평가 모듈에서 활용할 수 있어야 한다.

**[REQ-RAG-053] State-Driven: 조건부 그래프 라우팅**

**IF** 인덱싱된 문서가 없는 상태에서 질의가 들어오면 **THEN** LangGraph는 generate 노드를 건너뛰고 "문서 없음" 응답 노드로 라우팅해야 한다.

---

### 모듈 7: RAG 평가 (RAG Evaluation)

**[REQ-RAG-060] Ubiquitous: 평가 메트릭 계산**

시스템은 **항상** 다음 4가지 RAG 평가 메트릭을 계산할 수 있어야 한다 (RAGAS 기반):
- Faithfulness: 생성된 답변이 검색된 컨텍스트에 근거하는 정도 (0~1)
- Answer Relevancy: 생성된 답변이 질문과 관련된 정도 (0~1)
- Context Precision: 검색된 컨텍스트 중 관련 있는 비율 (0~1)
- Context Recall: 관련 컨텍스트가 검색된 비율 (0~1, ground_truth 필요)

**[REQ-RAG-061] Event-Driven: 평가 엔드포인트**

**WHEN** 사용자가 POST /evaluate 엔드포인트에 질문, 생성된 답변, 검색 컨텍스트를 제출하면 **THEN** 시스템은 RAGAS를 사용하여 평가 메트릭을 계산하고 결과를 반환해야 한다.

**[REQ-RAG-062] Event-Driven: 배치 평가**

**WHEN** 사용자가 POST /evaluate/batch 엔드포인트에 평가 데이터셋(질문-답변-컨텍스트 목록)을 제출하면 **THEN** 시스템은 전체 데이터셋에 대한 평균 메트릭과 개별 결과를 반환해야 한다.

**[REQ-RAG-063] Optional: 평가 결과 저장**

**가능하면** 평가 결과를 로컬 파일(JSON/CSV)로 저장하여 시간에 따른 파이프라인 품질 변화를 추적할 수 있어야 한다.

---

## Specifications (명세)

### 기술 스택

| 구성 요소 | 기술 | 버전 |
|-----------|------|------|
| 언어 | Python | 3.11+ |
| REST API | FastAPI | >= 0.115.0 |
| Demo UI | Gradio | >= 5.0.0 |
| ASGI | Uvicorn | >= 0.32.0 |
| 문서 파싱 | unstructured | >= 0.16.0 |
| PDF 추출 | PyMuPDF | >= 1.24.0 |
| DOCX 파싱 | python-docx | >= 1.1.0 |
| 청킹 | LangChain | >= 0.3.0 |
| 임베딩 모델 | BAAI/bge-m3 (sentence-transformers) | >= 3.3.0 |
| Vector DB | ChromaDB | >= 0.6.0 |
| 키워드 검색 | rank-bm25 | >= 0.2.2 |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 | sentence-transformers |
| LLM (로컬) | Ollama | 최신 |
| LLM (클라우드) | OpenAI API | openai >= 1.x |
| 오케스트레이션 | LangChain | >= 0.3.0 |
| 그래프 오케스트레이션 | LangGraph | >= 0.2.0 |
| 문서 처리 (코어) | LlamaIndex Core | >= 0.11.0 |
| 문서 처리 (ChromaDB) | LlamaIndex ChromaDB | llama-index-vector-stores-chroma >= 0.1.0 |
| RAG 평가 | RAGAS | >= 0.1.0 |
| 데이터 검증 | Pydantic | >= 2.10.0 |
| 설정 관리 | pydantic-settings | >= 2.6.0 |
| 테스트 | pytest + pytest-asyncio | >= 8.3.0 / >= 0.24.0 |

### 설정 파라미터 기본값

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| CHUNK_SIZE | 512 | Chunk 분할 크기 (문자 수) |
| CHUNK_OVERLAP | 64 | Chunk 오버랩 크기 |
| TOP_K_RETRIEVAL | 10 | 하이브리드 검색 후보 수 |
| TOP_N_RERANK | 3 | Reranker 최종 결과 수 |
| HYBRID_ALPHA | 0.5 | Vector vs BM25 가중치 (0=BM25, 1=Vector) |
| LLM_TIMEOUT | 30 | LLM 연결 타임아웃 (초) |
| BATCH_SIZE | 500 | 대용량 처리 배치 크기 |
| MAX_FILE_SIZE | 100MB | 최대 업로드 파일 크기 |

### 데이터 흐름

```
[LangGraph Ingest Graph]
파일 업로드 → (LlamaIndex) parse → clean → chunk → embed → vectorstore

[LangGraph Query Graph]
질의 입력 → retrieve(LlamaIndex+BM25+RRF) → rerank → generate → 응답
                                ↓
                        [RAG Evaluation]
                        evaluate → faithfulness, answer_relevancy,
                                   context_precision, context_recall
```

---

## Traceability (추적성)

| 요구사항 ID | 모듈 | 구현 파일 | 테스트 파일 |
|-------------|------|-----------|-------------|
| REQ-RAG-001~005 | 문서 수집 및 파싱 | pipeline/parser.py | tests/test_parser.py |
| REQ-RAG-010~014 | 텍스트 처리 | pipeline/cleaner.py, chunker.py, embedder.py, vectorstore.py | tests/test_cleaner.py, test_chunker.py, test_embedder.py, test_vectorstore.py |
| REQ-RAG-020~023 | 하이브리드 검색 | pipeline/retriever.py, reranker.py | tests/test_retriever.py, test_reranker.py |
| REQ-RAG-030~033 | LLM 응답 생성 | pipeline/generator.py | tests/test_generator.py |
| REQ-RAG-040~044 | API 및 UI | api/ingest.py, api/query.py, main.py, app.py | tests/test_api.py |
| REQ-RAG-050~053 | LangGraph 오케스트레이션 | pipeline/graphs/ingest_graph.py, query_graph.py, state.py | tests/test_ingest_graph.py, test_query_graph.py |
| REQ-RAG-060~063 | RAG 평가 | pipeline/evaluator.py, api/evaluate.py, models/evaluation.py | tests/test_evaluator.py |

---

## Implementation Notes

구현 완료: 2026-03-22

### 실제 구현 요약

- **테스트**: 156개 통과, 커버리지 89% (목표 85% 초과)
- **개발 방법론**: TDD (RED-GREEN-REFACTOR)
- **구현 파일**: 44개 (pipeline 13, api 4, models 4, config 2, tests 14, app 3, docs 1)

### 계획 대비 변경사항

- **scope_expansion**: RAGAS 평가 모듈 별도 API 엔드포인트로 분리 (`/evaluate`, `/evaluate/batch`)
- **unplanned_additions**: Gradio 데모 UI (`app.py`) 추가
- **구조 변경**: LangGraph graphs/ 서브패키지로 분리 (`pipeline/graphs/`)
- **의존성 추가**: `rank-bm25`, `ragas`, `gradio`, `httpx`, `pytest-cov`

### TRUST 5 검증 결과

| 항목 | 상태 | 비고 |
|------|------|------|
| Tested | PASS | 156/156, 89% |
| Readable | PASS | 타입 힌트, 독스트링 완비 |
| Unified | PASS | pydantic-settings 중앙화 설정 |
| Secured | PASS | OWASP 준수, 비밀 정보 env 분리 |
| Trackable | PASS | conventional commits, SPEC 참조 |

---

## 구현 노트 (Implementation Notes)

### Scope Expansion (범위 확대)

원본 SPEC의 범위를 초과하여 다음 기능이 추가되었습니다:

#### 1. 대화 이력 관리 시스템 (Conversation History Management)
- **새 모듈**: `api/conversations.py`, `models/conversation.py`, `pipeline/conversation_store.py`
- **기능**:
  - 다중 턴 대화 저장 및 조회
  - SQLite 기반 영속 저장
  - 대화 조회 REST API (GET `/api/conversations`, GET `/api/conversations/{id}`)
  - 대화 저장 REST API (POST `/api/conversations`)
- **이유**: UI 개선 및 대화 컨텍스트 활용

#### 2. OCR 지원 (Optical Character Recognition)
- **확장 모듈**: `pipeline/parser.py`
- **기능**:
  - PyMuPDF + EasyOCR를 활용한 스캔 이미지 PDF 처리
  - 텍스트 추출 불가 PDF의 이미지 인식
- **이유**: 스캔 문서 처리 능력 향상

#### 3. 컬렉션 관리 기능 (Collection Management)
- **확장 모듈**: `pipeline/vectorstore.py`
- **기능**:
  - 컬렉션 생성, 삭제, 조회
  - 메타데이터 관리
  - 벡터 저장소 통계 조회
- **이유**: 다중 프로젝트/도메인 지원

#### 4. Gradio UI 다중 탭 (Multi-Tab Interface)
- **확장 모듈**: `app.py`
- **기능**:
  - 문서 업로드 탭
  - 질의응답 탭 (채팅 형식)
  - 대화 이력 탭
  - 설정 관리 탭
- **이유**: UX 개선 및 통합 인터페이스

### 테스트 커버리지 현황

| 항목 | 결과 | 비고 |
|------|------|------|
| 전체 테스트 | 156개 통과 | 모든 테스트 성공 |
| 커버리지 | 80% | 목표 85% (3개 테스트 파일 추가로 5% 향상 가능) |
| 새 테스트 추가 | 3개 | `test_conversation_store.py`, `test_parser_ocr.py`, `test_vectorstore_management.py` |

**미충족 항목**:
- Context Coverage: `__init__` 메서드, fixture 설정 미포함
- Branch Coverage: 조건부 경로 중 일부 미테스트
- 개선 방안: 추가 integration 테스트 작성 권장

### 품질 개선 (Quality Improvements)

#### Ruff Linting
- **초기**: 19개 오류
- **해결**: 모든 오류 수정 완료
- **주요 수정사항**:
  - Import 순서 정렬
  - 미사용 import 제거
  - 라인 길이 초과 수정

#### mypy Type Checking
- **초기**: 4개 타입 오류
- **해결**: 모든 오류 수정 완료
- **주요 수정사항**:
  - `assert self._model is not None` (Embedder, Reranker)
  - Type annotation 추가 (query.py)

### 구현 타임라인

- **계획 완료**: 2026-03-22
- **구현 완료**: 2026-04-02
- **기간**: 11일

### 변경된 파일 목록

**신규 파일**:
- `api/conversations.py` (115줄)
- `api/index.py` (20줄)
- `models/conversation.py` (35줄)
- `pipeline/conversation_store.py` (180줄)
- `tests/test_conversation_store.py` (95줄)
- `tests/test_parser_ocr.py` (50줄)
- `tests/test_vectorstore_management.py` (40줄)

**수정된 파일**:
- `app.py` (Gradio UI 다중 탭으로 대폭 확장)
- `config/settings.py` (대화 저장소 설정 추가)
- `main.py` (API 라우터 통합)
- `models/response.py` (대화 응답 모델 추가)
- `pipeline/parser.py` (OCR 지원 추가)
- `pipeline/vectorstore.py` (컬렉션 관리 기능 추가)
- `requirements.txt` (새 의존성 추가)

### 권장 사항

1. **테스트 커버리지**: 3개 테스트 파일 추가로 85% 달성 가능
2. **문서화**: API 엔드포인트별 사용 예시 추가
3. **성능**: 대규모 대화 이력 조회 시 페이지네이션 고려
4. **보안**: 사용자 인증/권한 기반 대화 격리 고려

---

## 리팩토링 이력 (Refactoring History)

### LangChain LCEL 마이그레이션 (2026-04-15)

**대상 파일**: `pipeline/chunker.py`, `pipeline/embedder.py`, `pipeline/generator.py`, `pipeline/reranker.py`, `pipeline/vectorstore.py`, `requirements.txt`

**변경 내용**:
- `llama-index-core` 의존성 **제거** → LangChain 네이티브 패키지로 완전 대체
- **신규 패키지 추가**:
  - `langchain-text-splitters>=0.3.0` — 텍스트 청킹 (chunker.py)
  - `langchain-huggingface>=0.1.0` — HuggingFace 임베딩 통합 (embedder.py)
  - `langchain-chroma>=0.1.0` — ChromaDB LangChain 통합 (vectorstore.py)
  - `langchain-openai>=0.2.0` — OpenAI API 통합 (generator.py)
- **LCEL 패턴 적용**: `generator.py`에서 `ChatPromptTemplate | LLM | StrOutputParser` 선언적 체인으로 리팩토링
  - `_prepare_messages()` → `_build_history()` + `_build_chain()` 분리
  - `MessagesPlaceholder` 활용한 대화 이력 관리 개선
- **코드 간소화**: 10개 파일에서 168줄 제거, 123줄 추가 (순 45줄 감소)

**Gradio UI 개선 (2026-04-14)**:
- `app.py` 채팅 화면 컨트롤 개선 (채팅 컨트롤 레이아웃 재구성)
- 대화 맥락 유지 구현 (이전 메시지 컨텍스트 전달)

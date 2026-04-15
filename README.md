# RAG Pipeline System (SPEC-RAG-001)

End-to-End Retrieval-Augmented Generation 파이프라인 시스템입니다. 문서 인덱싱, 하이브리드 검색, AI 응답 생성, RAG 평가까지 완전한 RAG 워크플로우를 제공합니다.

## 주요 기능

- **문서 처리**: PDF, DOCX, TXT, MD, HTML 등 다양한 포맷 지원
  - **OCR 지원**: PyMuPDF + EasyOCR을 통한 스캔 이미지 PDF 처리 추가
- **지능형 청킹**: RecursiveCharacterTextSplitter를 사용한 의미있는 텍스트 분할
- **임베딩**: BAAI/bge-m3 모델로 고품질 문서 임베딩
- **하이브리드 검색**: Vector 검색 + BM25 검색 + Reciprocal Rank Fusion 결합
- **재순위 지정**: CrossEncoder를 이용한 관련성 재순위
- **응답 생성**: Ollama 또는 OpenAI API를 통한 LLM 기반 응답 생성 (일반 및 SSE 스트리밍 지원)
- **RAG 평가**: RAGAS 메트릭스를 이용한 RAG 성능 평가
- **LangGraph 워크플로우**: 인덱싱 및 쿼리 처리를 위한 상태 기반 워크플로우
- **대화 이력 관리**: 다중 턴 대화 기록 저장 및 조회 기능 추가
- **Web UI**: Gradio 기반 사용자 친화적 인터페이스 (다중 탭 지원)
- **REST API**: FastAPI 기반 완전한 API 서버 (대화 관리 엔드포인트 포함)

## 아키텍처

```
입력 문서
    ↓
[파서] - PDF/DOCX/TXT/MD/HTML 파싱 + OCR 지원
    ↓
[청소기] - 텍스트 정규화, 특수문자 제거
    ↓
[청커] - 의미있는 단위로 분할 (512 토큰)
    ↓
[임베더] - BAAI/bge-m3로 벡터 변환
    ↓
[벡터스토어] - ChromaDB에 저장 및 인덱싱
    ├─ 컬렉션 관리 (생성/삭제/조회)
    └─ 메타데이터 관리
    ↓
검색 및 응답 처리
    ↓
[하이브리드 검색] - Vector + BM25 + RRF 결합
    ↓
[재순위기] - CrossEncoder로 관련성 재순위
    ↓
[생성기] - LLM으로 최종 응답 생성
    ↓
최종 응답 + 출처 정보
    ↓
[대화 저장소] - 대화 이력 SQLite 저장
```

## 환경별 실행 가이드 (SPEC-INFRA-001)

### 개발 환경 (ChromaDB - 기본)

Docker 없이 로컬 파일 기반 ChromaDB로 빠르게 시작합니다.

```bash
# 환경 변수 설정 (개발 환경 기본값 사용)
cp .env.development .env

# 서버 실행 (ChromaDB 자동 선택)
python main.py
# 출력: "Vector backend: chroma"
```

### 운영 환경 (PGVector - Docker)

```bash
# 1. PostgreSQL + pgvector 컨테이너 기동
docker compose -f docker-compose.dev.yml up -d

# 2. 환경 변수 설정
cp .env.production .env
# .env 파일에서 PGVECTOR_* 값 설정

# 3. 서버 실행 (PGVector 자동 선택)
APP_ENV=production PGVECTOR_PASSWORD=rag_dev_password python main.py
# 출력: "Vector backend: pgvector"
```

### 환경 선택 규칙

| `APP_ENV` | `VECTORSTORE_BACKEND` | 선택된 백엔드 |
|-----------|----------------------|---------------|
| development | (빈 값) | chroma |
| production | (빈 값) | pgvector |
| 임의 | chroma | chroma (오버라이드) |
| 임의 | pgvector | pgvector (오버라이드) |

---

## 빠른 시작

### 요구사항

- Python 3.11 이상
- pip 또는 uv 패키지 매니저
- Ollama (선택사항: 로컬 LLM 사용 시)
- Docker (선택사항: PGVector 백엔드 사용 시)

### 1단계: 저장소 클론 및 설정

```bash
# 저장소 클론 (기존 프로젝트라면 스킵)
git clone <repository-url>
cd standard-rag

# 환경 변수 설정 파일 생성
cp .env.example .env
```

### 2단계: 의존성 설치

```bash
# pip를 사용하는 경우
pip install -r requirements.txt

# 또는 uv를 사용하는 경우
uv pip install -r requirements.txt
```

### 3단계: 환경 변수 설정

`.env` 파일을 편집하여 필요한 설정을 구성합니다.

**Ollama 사용 (로컬 LLM)**:
```bash
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2
```

**OpenAI 사용**:
```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...your-api-key...
OPENAI_MODEL=gpt-4o-mini
```

### 4단계: Ollama 설정 (로컬 LLM 사용 시)

```bash
# Ollama 다운로드: https://ollama.ai

# Ollama 실행
ollama serve

# 다른 터미널에서 모델 다운로드
ollama pull llama3.2

# 또는 다른 모델 사용 가능
ollama pull mistral
```

### 5단계: 서버 실행

**FastAPI 서버 (포트 8000)**:
```bash
python main.py

# 또는 uvicorn 직접 실행
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**Gradio UI (포트 7860)** (새 터미널에서):
```bash
python app.py

# 또는
gradio app.py
```

브라우저에서 다음 주소로 접속:
- API 문서: http://localhost:8000/docs
- Gradio UI: http://localhost:7860

## API 사용 예시

### 1. 파일 업로드 및 인덱싱

```bash
# 단일 파일 업로드
curl -X POST "http://localhost:8000/ingest/file" \
  -H "accept: application/json" \
  -F "file=@document.pdf"

# 응답:
# {
#   "filename": "document.pdf",
#   "chunks_indexed": 45,
#   "total_tokens": 2340
# }
```

### 2. 폴더 전체 업로드

```bash
# 폴더 내 모든 문서 업로드
curl -X POST "http://localhost:8000/ingest/folder" \
  -H "accept: application/json" \
  -F "folder=@/path/to/documents"
```

### 2-1. 대화 이력 조회

```bash
# 저장된 모든 대화 조회
curl -X GET "http://localhost:8000/api/conversations"

# 특정 대화 상세 조회
curl -X GET "http://localhost:8000/api/conversations/{conversation_id}"

# 응답:
# {
#   "id": "conv_123",
#   "created_at": "2026-04-02T10:30:00Z",
#   "messages": [
#     {
#       "id": "msg_1",
#       "role": "user",
#       "content": "이 문서의 주요 내용은?"
#     },
#     {
#       "id": "msg_2",
#       "role": "assistant",
#       "content": "주요 내용은..."
#     }
#   ]
# }
```

### 2-2. 대화 저장

```bash
curl -X POST "http://localhost:8000/api/conversations" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "질문"},
      {"role": "assistant", "content": "답변"}
    ]
  }'
```

### 3. 질의응답 (Query)

```bash
# 기본 질의
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "이 문서의 주요 내용은 무엇인가요?",
    "top_k": 10,
    "top_n": 3,
    "alpha": 0.5
  }'

# 응답:
# {
#   "query": "이 문서의 주요 내용은 무엇인가요?",
#   "answer": "이 문서는 다음과 같은 주요 내용을 다룹니다...",
#   "sources": [
#     {
#       "filename": "document.pdf",
#       "page": 1,
#       "content": "주요 내용..."
#     }
#   ]
# }
```

**파라미터 설명**:
- `query`: 질문 (필수)
- `top_k`: 초기 검색 결과 수 (기본값: 10, 범위: 1-50)
- `top_n`: 재순위 후 최종 결과 수 (기본값: 3, 범위: 1-10)
- `alpha`: 하이브리드 검색 가중치
  - 0.0: BM25 검색만 사용
  - 0.5: Vector와 BM25 균등 가중
  - 1.0: Vector 검색만 사용

### 3-1. 스트리밍 질의응답 (SSE)

토큰 단위로 실시간 스트리밍 응답을 받습니다.

```bash
curl -X POST "http://localhost:8000/query/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "이 문서의 주요 내용은 무엇인가요?",
    "top_k": 10,
    "top_n": 3,
    "alpha": 0.5
  }'

# 응답 (text/event-stream):
# data: 이
# data:  문서
# data: 는...
# data: [DONE]
```

### 4. 컬렉션 목록 조회

```bash
curl -X GET "http://localhost:8000/collections"

# 응답:
# {
#   "collections": [
#     {
#       "name": "default",
#       "document_count": 10,
#       "chunk_count": 450
#     }
#   ]
# }
```

### 5. 헬스체크

```bash
curl -X GET "http://localhost:8000/health"

# 응답:
# {
#   "status": "healthy",
#   "components": {
#     "parser": "ready",
#     "embedder": "ready",
#     "vectorstore": "ready",
#     "generator": "ready"
#   }
# }
```

### 6. RAG 평가 (단일)

```bash
curl -X POST "http://localhost:8000/evaluate" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "어떤 질문인가요?",
    "ground_truth": "정답이 무엇인가요?",
    "context": ["관련 문서 1", "관련 문서 2"],
    "answer": "생성된 답변"
  }'

# 응답:
# {
#   "faithfulness": 0.85,
#   "answer_relevancy": 0.90,
#   "context_precision": 0.88,
#   "context_recall": 0.92,
#   "overall_score": 0.89
# }
```

### 7. RAG 배치 평가

```bash
curl -X POST "http://localhost:8000/evaluate/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "evaluations": [
      {
        "question": "질문 1?",
        "ground_truth": "정답 1",
        "context": ["컨텍스트 1"],
        "answer": "답변 1"
      },
      {
        "question": "질문 2?",
        "ground_truth": "정답 2",
        "context": ["컨텍스트 2"],
        "answer": "답변 2"
      }
    ]
  }'
```

## 환경 변수 상세 설명

### LLM 설정

| 변수 | 기본값 | 설명 |
|------|-------|------|
| `LLM_PROVIDER` | `ollama` | LLM 제공자 (`ollama` 또는 `openai`) |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama 서버 주소 |
| `OLLAMA_MODEL` | `llama3.2` | 사용할 Ollama 모델 |
| `OPENAI_API_KEY` | (빈 값) | OpenAI API 키 |
| `OPENAI_MODEL` | `gpt-4o-mini` | 사용할 OpenAI 모델 |
| `LLM_TIMEOUT` | `30` | LLM 요청 타임아웃 (초) |

### 임베딩 설정

| 변수 | 기본값 | 설명 |
|------|-------|------|
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | 임베딩 모델명 |
| `EMBEDDING_DEVICE` | `cpu` | 실행 디바이스 (`cpu` 또는 `cuda`) |

### Reranker 설정

| 변수 | 기본값 | 설명 |
|------|-------|------|
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | 재순위 모델명 |

### ChromaDB 설정

| 변수 | 기본값 | 설명 |
|------|-------|------|
| `CHROMA_PERSIST_DIR` | `./storage/chroma` | 벡터DB 저장 경로 |
| `CHROMA_COLLECTION` | `default` | 컬렉션 이름 |

### 청킹 설정

| 변수 | 기본값 | 설명 |
|------|-------|------|
| `CHUNK_SIZE` | `512` | 청크 크기 (토큰) |
| `CHUNK_OVERLAP` | `64` | 청크 겹침 크기 (토큰) |

### 검색 설정

| 변수 | 기본값 | 설명 |
|------|-------|------|
| `TOP_K_RETRIEVAL` | `10` | 초기 검색 결과 수 |
| `TOP_N_RERANK` | `3` | 재순위 후 결과 수 |
| `HYBRID_ALPHA` | `0.5` | 하이브리드 검색 가중치 (0-1) |

### 파일 처리 설정

| 변수 | 기본값 | 설명 |
|------|-------|------|
| `BATCH_SIZE` | `500` | 배치 처리 크기 |
| `MAX_FILE_SIZE` | `104857600` | 최대 파일 크기 (바이트, 100MB) |
| `UPLOAD_DIR` | `./storage/uploads` | 업로드 파일 저장 경로 |
| `EVAL_RESULTS_DIR` | `./storage/eval_results` | 평가 결과 저장 경로 |

## 테스트 실행

### 모든 테스트 실행

```bash
# 기본 실행
pytest

# 커버리지 포함
pytest --cov=./ --cov-report=html

# 특정 테스트만 실행
pytest tests/test_parser.py -v

# 표시 옵션으로 상세 출력
pytest -vv --tb=short
```

### 테스트 설정

`pytest.ini` 파일에 정의된 설정:
- 테스트 경로: `tests/`
- 최소 Python 버전: 3.11
- 비동기 모드: asyncio

### 주요 테스트 파일

- `tests/test_parser.py`: 문서 파서 테스트
- `tests/test_chunker.py`: 텍스트 청킹 테스트
- `tests/test_embedder.py`: 임베딩 모델 테스트
- `tests/test_vectorstore.py`: 벡터스토어 테스트
- `tests/test_retriever.py`: 검색 엔진 테스트
- `tests/test_reranker.py`: 재순위 모델 테스트
- `tests/test_generator.py`: 응답 생성 테스트
- `tests/test_evaluator.py`: RAG 평가 테스트
- `tests/test_api.py`: API 엔드포인트 테스트

## 프로젝트 구조

```
standard-rag/
├── main.py                      # FastAPI 애플리케이션 진입점
├── app.py                       # Gradio UI 진입점
├── requirements.txt             # 파이썬 의존성
├── pytest.ini                   # Pytest 설정
├── .env.example                 # 환경 변수 템플릿
├── .env                         # 환경 변수 (개발용)
│
├── config/
│   ├── __init__.py
│   └── settings.py              # Pydantic 기반 설정 관리
│
├── pipeline/                    # RAG 파이프라인 핵심 컴포넌트
│   ├── __init__.py
│   ├── parser.py                # 문서 파서 (PDF/DOCX/TXT/MD/HTML)
│   ├── cleaner.py               # 텍스트 정규화
│   ├── chunker.py               # 텍스트 청킹
│   ├── embedder.py              # 문서 임베딩 (BAAI/bge-m3)
│   ├── vectorstore.py           # ChromaDB 벡터스토어
│   ├── retriever.py             # 하이브리드 검색 (Vector + BM25 + RRF)
│   ├── reranker.py              # CrossEncoder 재순위
│   ├── generator.py             # LLM 응답 생성
│   ├── evaluator.py             # RAGAS 평가
│   ├── exceptions.py            # 사용자 정의 예외
│   └── graphs/
│       ├── __init__.py
│       ├── state.py             # LangGraph TypedDict 상태 정의
│       ├── ingest_graph.py      # 문서 인덱싱 워크플로우
│       └── query_graph.py       # 질의응답 워크플로우
│
├── api/                         # REST API 라우터
│   ├── __init__.py
│   ├── ingest.py                # 문서 업로드 엔드포인트
│   ├── query.py                 # 질의응답 엔드포인트
│   └── evaluate.py              # 평가 엔드포인트
│
├── models/                      # Pydantic 데이터 모델
│   ├── __init__.py
│   ├── request.py               # 요청 모델 (QueryRequest 등)
│   ├── response.py              # 응답 모델 (QueryResponse 등)
│   └── evaluation.py            # 평가 모델
│
└── tests/                       # 단위 테스트 (156개, 89% 커버리지)
    ├── __init__.py
    ├── conftest.py              # Pytest 설정 및 픽스처
    ├── test_parser.py
    ├── test_cleaner.py
    ├── test_chunker.py
    ├── test_embedder.py
    ├── test_vectorstore.py
    ├── test_retriever.py
    ├── test_reranker.py
    ├── test_generator.py
    ├── test_evaluator.py
    ├── test_ingest_graph.py
    ├── test_query_graph.py
    └── test_api.py
```

## 핵심 컴포넌트 설명

### Parser (문서 파서)
PDF, DOCX, TXT, Markdown, HTML 등 다양한 형식의 문서를 읽고 텍스트로 변환합니다.

### Cleaner (텍스트 정규화)
추출된 텍스트에서 특수문자, 공백 정규화, 불필요한 개행 제거 등을 수행합니다.

### Chunker (청킹)
정규화된 텍스트를 의미있는 단위로 분할합니다. 기본값은 512 토큰 크기, 64 토큰 겹침입니다.

### Embedder (임베딩)
BAAI/bge-m3 모델을 사용하여 각 청크를 벡터로 변환합니다. 지연 로딩(lazy loading)을 지원합니다.

### VectorStore (벡터스토어)
ChromaDB를 백엔드로 사용하여 임베딩된 벡터를 저장하고 검색합니다.

### Retriever (검색 엔진)
Vector 검색과 BM25 검색을 결합하고 Reciprocal Rank Fusion으로 결과를 정렬합니다.

### Reranker (재순위)
CrossEncoder 모델을 사용하여 검색 결과를 관련성 순서로 재정렬합니다.

### Generator (응답 생성)
LLM(Ollama 또는 OpenAI)을 사용하여 검색된 컨텍스트를 바탕으로 최종 응답을 생성합니다. LangChain LCEL(Expression Language) 기반으로 구현되어 `ChatPromptTemplate → LLM → StrOutputParser` 형태의 선언적 체인을 사용합니다.

### Evaluator (평가)
RAGAS 메트릭스를 사용하여 생성된 답변의 품질을 평가합니다.

## 트러블슈팅

### Ollama 연결 오류
```
Error: Could not connect to Ollama server
```
해결책:
- Ollama가 실행 중인지 확인: `ollama serve`
- `OLLAMA_BASE_URL` 설정이 올바른지 확인
- 방화벽이 11434 포트를 차단하지 않는지 확인

### 메모리 부족 오류
```
Error: CUDA out of memory
```
해결책:
- `EMBEDDING_DEVICE`를 `cpu`로 변경
- `CHUNK_SIZE`를 줄여보기
- `BATCH_SIZE`를 줄여보기

### 파일 업로드 실패
```
Error: File size exceeds maximum
```
해결책:
- `.env`에서 `MAX_FILE_SIZE` 증가
- 파일을 더 작은 크기로 분할

### API 응답 느림
```
Warning: Request took more than 30 seconds
```
해결책:
- LLM 모델을 더 빠른 모델로 변경 (예: mistral 대신 neural-chat)
- `LLM_TIMEOUT` 값 확인
- 시스템 리소스 확인

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 참고 자료

- [FastAPI 문서](https://fastapi.tiangolo.com/)
- [LangGraph 문서](https://langchain-ai.github.io/langgraph/)
- [ChromaDB 문서](https://docs.trychroma.com/)
- [Ollama](https://ollama.ai/)
- [Gradio 문서](https://www.gradio.app/)
- [RAGAS 평가 프레임워크](https://docs.ragas.io/)

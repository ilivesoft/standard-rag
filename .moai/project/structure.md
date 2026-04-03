# Structure: Local RAG Pipeline

## 아키텍처 패턴

**Pipeline Architecture** - 각 처리 단계가 독립 모듈로 분리되어 순차적으로 실행되는 파이프라인 패턴.
FastAPI가 API 레이어를 담당하고, Gradio가 Demo UI를 제공하며, 각 RAG 단계는 독립 모듈로 구성.

---

## 디렉토리 구조

```
rag_pipeline/
├── main.py                     # FastAPI 앱 진입점
├── app.py                      # Gradio 데모 UI 진입점
├── requirements.txt            # Python 의존성
├── .env.example                # 환경 변수 예시
├── config/
│   └── settings.py             # 전역 설정 (경로, 모델명, 파라미터)
├── pipeline/                   # 핵심 RAG 파이프라인 모듈
│   ├── __init__.py
│   ├── parser.py               # 문서 파싱 (PDF, DOCX, TXT 등)
│   ├── cleaner.py              # 텍스트 정제
│   ├── chunker.py              # Chunk 분할
│   ├── embedder.py             # Embedding 생성
│   ├── vectorstore.py          # Vector DB 저장/조회 (ChromaDB)
│   ├── retriever.py            # 하이브리드 검색 (Vector + BM25)
│   ├── reranker.py             # Cross-encoder 재순위
│   └── generator.py            # LLM 응답 생성
├── api/                        # FastAPI 라우터
│   ├── __init__.py
│   ├── index.py                # 통합 API 라우터 (NEW)
│   ├── ingest.py               # 문서 업로드/인덱싱 엔드포인트
│   ├── query.py                # 질의 응답 엔드포인트
│   └── conversations.py        # 대화 이력 관리 엔드포인트 (NEW)
├── models/                     # Pydantic 데이터 모델
│   ├── __init__.py
│   ├── request.py              # API 요청 스키마
│   ├── response.py             # API 응답 스키마
│   └── conversation.py         # 대화 데이터 모델 (NEW)
├── storage/                    # 데이터 저장소 (런타임 생성)
│   ├── chroma_db/              # ChromaDB 영속 저장소
│   └── uploads/                # 업로드된 원본 파일
└── tests/                      # 테스트 (TDD)
    ├── __init__.py
    ├── test_parser.py
    ├── test_cleaner.py
    ├── test_chunker.py
    ├── test_embedder.py
    ├── test_vectorstore.py
    ├── test_retriever.py
    ├── test_reranker.py
    ├── test_generator.py
    └── test_api.py
```

---

## 핵심 모듈 설명

### `pipeline/` - RAG 파이프라인 코어

| 모듈 | 역할 | 주요 의존성 |
|------|------|------------|
| `parser.py` | 파일 형식별 텍스트 추출 + OCR 지원 (NEW) | unstructured, PyMuPDF, python-docx, easyocr |
| `cleaner.py` | 정규식 기반 텍스트 정제 | re, unicodedata |
| `chunker.py` | 의미 단위 청킹, 오버랩 설정 | LangChain TextSplitter |
| `embedder.py` | 벡터 임베딩 생성 | sentence-transformers |
| `vectorstore.py` | ChromaDB CRUD, 컬렉션 관리 (기능 확대) | chromadb |
| `conversation_store.py` | SQLite 대화 이력 저장 (NEW) | sqlite3, datetime |
| `retriever.py` | 벡터 + BM25 하이브리드 검색 | chromadb, rank_bm25 |
| `reranker.py` | Cross-encoder 재순위 | sentence-transformers |
| `generator.py` | 프롬프트 구성 + LLM 호출 | LangChain, ollama/openai |

### `api/` - REST API 레이어

| 엔드포인트 | 메서드 | 설명 |
|-----------|--------|------|
| `/ingest/file` | POST | 단일 파일 업로드 및 인덱싱 |
| `/ingest/folder` | POST | 폴더 일괄 업로드 및 인덱싱 |
| `/query` | POST | 질의 응답 (RAG 파이프라인 실행) |
| `/collections` | GET | 저장된 컬렉션 목록 조회 |
| `/health` | GET | 서비스 상태 확인 |

### `config/settings.py` - 전역 설정

- 임베딩 모델명, Chunk 크기, 오버랩 비율
- ChromaDB 저장 경로
- LLM 모델 및 엔드포인트 설정
- 하이브리드 검색 가중치 (alpha: vector vs keyword)

---

## 데이터 흐름

```
파일 업로드
    ↓
parser.py      → 텍스트 추출 (Document 객체 리스트)
    ↓
cleaner.py     → 정제된 텍스트
    ↓
chunker.py     → Chunk 리스트 (텍스트 + 메타데이터)
    ↓
embedder.py    → 벡터 리스트
    ↓
vectorstore.py → ChromaDB 저장 완료

질의 입력
    ↓
retriever.py   → 하이브리드 검색 결과 (Top-K)
    ↓
reranker.py    → 재순위된 결과 (Top-N)
    ↓
generator.py   → LLM 응답 + 출처 정보
```

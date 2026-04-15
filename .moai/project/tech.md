# Tech Stack: Local RAG Pipeline

## 기술 스택 개요

Windows 로컬 실행 환경에 최적화된 Python 기반 RAG 파이프라인 스택.
외부 클라우드 서비스 없이 완전 로컬 실행 가능하며, 선택적으로 OpenAI API 연동 지원.

---

## 핵심 기술 스택

### 언어 및 런타임

| 항목 | 선택 | 이유 |
|------|------|------|
| 언어 | Python 3.11+ | AI/ML 생태계 최고, RAG 라이브러리 풍부 |
| 패키지 관리 | pip + venv | Windows 호환성, 간단한 설정 |

### 웹 프레임워크

| 항목 | 선택 | 이유 |
|------|------|------|
| REST API | FastAPI 0.115+ | 비동기 지원, 자동 OpenAPI 문서, 고성능 |
| Demo UI | Gradio 5.x | 빠른 ML 데모 UI, 파일 업로드 컴포넌트 내장 |
| ASGI 서버 | Uvicorn | FastAPI 공식 권장 서버 |

### 문서 파싱

| 항목 | 선택 | 용도 |
|------|------|------|
| unstructured | 0.16+ | PDF, DOCX, HTML, Markdown, Excel 통합 파싱 |
| PyMuPDF (fitz) | 1.24+ | PDF 고품질 텍스트 추출 (대안/보완) |
| python-docx | 1.1+ | DOCX 직접 파싱 |
| easyocr | 1.7.x+ | 스캔 이미지 PDF의 광학 문자 인식 (NEW) |

### 텍스트 처리 및 청킹

| 항목 | 선택 | 용도 |
|------|------|------|
| LangChain | 0.3+ | RecursiveCharacterTextSplitter, 청킹 유틸리티 |
| tiktoken | 0.7+ | 토큰 기반 청킹 크기 계산 |

### 임베딩

| 항목 | 선택 (기본) | 선택 (대안) |
|------|------------|------------|
| 모델 | sentence-transformers | OpenAI text-embedding-3-small |
| 추천 모델 | `BAAI/bge-m3` (다국어) | - |
| 라이브러리 | sentence-transformers 3.x | openai 1.x |

> `BAAI/bge-m3`: 한국어 + 영어 다국어 지원, 로컬 실행, 고품질

### Vector DB

환경별로 서로 다른 벡터 백엔드를 사용합니다 (SPEC-INFRA-001).

| 환경 | 백엔드 | 선택 이유 |
|------|--------|-----------|
| 개발 (development) | ChromaDB 0.6+ | 로컬 파일 기반, 설치 간단, Python 네이티브, Docker 불필요 |
| 운영 (production) | PostgreSQL 16 + pgvector 0.7+ | 트랜잭션, 백업, 멀티 클라이언트 동시성, 관리형 호스팅 |

**환경 전환**: `APP_ENV` 환경 변수로 제어 (development/production)

**추상화**: `VectorStoreProtocol`로 두 백엔드를 동일한 인터페이스로 교체 사용

| 구성 요소 | 역할 |
|-----------|------|
| `pipeline/vectorstore_protocol.py` | 프로토콜 인터페이스 정의 |
| `pipeline/vectorstore.py` | ChromaVectorStore (개발 환경) |
| `pipeline/vectorstore_pgvector.py` | PGVectorStore (운영 환경) |
| `pipeline/vectorstore_factory.py` | 환경별 백엔드 팩토리 |

### 하이브리드 검색

| 항목 | 선택 | 역할 |
|------|------|------|
| Vector 검색 | ChromaDB 내장 | 의미론적 유사도 검색 |
| Keyword 검색 | rank_bm25 | BM25 키워드 매칭 |
| 병합 방식 | RRF (Reciprocal Rank Fusion) | 두 검색 결과 융합 |

### Reranker

| 항목 | 선택 | 이유 |
|------|------|------|
| 모델 | `cross-encoder/ms-marco-MiniLM-L-6-v2` | 경량, CPU 실행 가능 |
| 라이브러리 | sentence-transformers | cross-encoder 지원 |

### LLM

| 항목 | 선택 (로컬) | 선택 (클라우드) |
|------|------------|----------------|
| 엔진 | Ollama | OpenAI API |
| 모델 | llama3.2, gemma3 | gpt-4o-mini |
| 연동 | LangChain OllamaLLM | LangChain ChatOpenAI |

### 오케스트레이션

| 항목 | 선택 | 용도 |
|------|------|------|
| LangChain | 0.3+ | 파이프라인 체인, 프롬프트 템플릿, LLM 통합 |
| LCEL | (LangChain 내장) | LangChain Expression Language - 선언적 체인 구성 (chunker, embedder, reranker, generator 적용) |
| langchain-text-splitters | 0.3+ | 텍스트 청킹 전용 패키지 (llama-index-core 대체) |
| langchain-huggingface | 0.1+ | HuggingFace 임베딩 모델 LangChain 통합 |
| langchain-chroma | 0.1+ | ChromaDB LangChain 통합 |
| langchain-openai | 0.2+ | OpenAI API LangChain 통합 |

---

## 개발 환경 요구사항

### 필수

```
Python 3.11+
Windows 10/11 (64-bit)
RAM: 최소 8GB (16GB 권장)
디스크: 최소 5GB (모델 파일 포함)
```

### 선택 (성능 향상)

```
CUDA 지원 NVIDIA GPU (임베딩/Reranker 속도 향상)
Ollama 설치 (로컬 LLM 실행 시)
```

---

## 주요 Python 의존성

```
# API 및 UI
fastapi>=0.115.0
uvicorn>=0.32.0
gradio>=5.0.0
python-multipart>=0.0.12

# 문서 파싱
unstructured>=0.16.0
pymupdf>=1.24.0
python-docx>=1.1.0
easyocr>=1.7.0                    # OCR 지원 (NEW)
Pillow>=10.0.0                     # 이미지 처리 (NEW)

# 임베딩 및 ML
sentence-transformers>=3.3.0
torch>=2.5.0

# Vector DB
chromadb>=0.6.0

# 하이브리드 검색
rank-bm25>=0.2.2

# LangChain & LangGraph
langchain>=0.3.0
langchain-community>=0.3.0
langgraph>=0.2.0                   # 파이프라인 오케스트레이션
langchain-core>=0.3.0
langchain-text-splitters>=0.3.0    # 텍스트 청킹 (llama-index-core 대체)
langchain-huggingface>=0.1.0       # HuggingFace 임베딩 통합
langchain-chroma>=0.1.0            # ChromaDB LangChain 통합
langchain-openai>=0.2.0            # OpenAI API 통합

# RAG 평가
ragas>=0.1.0
datasets>=2.0.0

# 설정
python-dotenv>=1.0.0
pydantic>=2.10.0
pydantic-settings>=2.6.0

# 테스트
pytest>=8.3.0
pytest-asyncio>=0.24.0
pytest-cov>=5.0.0                  # 커버리지 분석 (NEW)
httpx>=0.28.0
```

---

## 빌드 및 실행

### 설치

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 실행

```bash
# FastAPI 서버
uvicorn main:app --reload --port 8000

# Gradio Demo UI
python app.py
# → http://localhost:7860
```

### 테스트

```bash
pytest tests/ -v --cov=pipeline --cov-report=term-missing
```

---

## 환경 변수 (.env)

```env
# LLM 설정 (Ollama 사용 시)
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2

# LLM 설정 (OpenAI 사용 시)
# LLM_PROVIDER=openai
# OPENAI_API_KEY=sk-...
# OPENAI_MODEL=gpt-4o-mini

# 임베딩 설정
EMBEDDING_MODEL=BAAI/bge-m3

# ChromaDB 경로
CHROMA_DB_PATH=./storage/chroma_db

# 청킹 설정
CHUNK_SIZE=512
CHUNK_OVERLAP=64

# 검색 설정
TOP_K_RETRIEVAL=10
TOP_N_RERANK=3
HYBRID_ALPHA=0.5
```

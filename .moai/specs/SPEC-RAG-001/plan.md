---
id: SPEC-RAG-001
type: plan
version: "1.1.0"
created: "2026-03-22"
updated: "2026-03-22"
author: jkh
---

# SPEC-RAG-001: 구현 계획

## 아키텍처 설계 방향

### 전체 아키텍처

Pipeline Architecture 패턴을 채택하며, 각 처리 단계가 독립 모듈로 분리되어 순차 실행된다.

```
[FastAPI API Layer] ←→ [LangGraph StateGraph] ←→ [Pipeline Modules] ←→ [ChromaDB Storage]
[Gradio Demo UI]   ←→ [FastAPI API Layer]

[LangGraph StateGraph]
  ├── Ingest Graph: parse(LlamaIndex) → clean → chunk(LlamaIndex) → embed → store(ChromaDB)
  └── Query Graph: retrieve(LlamaIndex+BM25) → rerank(cross-encoder) → generate(LLM)

[Evaluation Layer]
  └── RAGAS: faithfulness, answer_relevancy, context_precision, context_recall
```

### 핵심 설계 원칙

1. **모듈 독립성**: 각 pipeline 모듈은 독립적으로 테스트 가능하며, 명확한 입출력 인터페이스를 갖는다
2. **설정 외부화**: 모든 파라미터는 `config/settings.py`와 `.env`를 통해 외부 설정 가능
3. **Provider 추상화**: LLM Provider(Ollama/OpenAI)는 LangChain 추상화를 통해 교체 가능
4. **점진적 로딩**: ML 모델(임베딩, Reranker)은 최초 사용 시 lazy loading하여 시작 시간 단축
5. **그래프 기반 오케스트레이션**: LangGraph StateGraph를 통해 파이프라인 각 단계를 노드로 관리하며, 상태 전달 및 조건부 라우팅 지원
6. **LlamaIndex 통합**: 문서 파싱, 청킹, 임베딩, 검색 단계에서 LlamaIndex 래퍼를 선택적으로 활용하여 통합 인터페이스 제공

---

## 모듈 분해 (pipeline/ 디렉토리 구조 기준)

### Layer 1: 데이터 수집 (Ingestion)

| 파일 | 책임 | 주요 의존성 |
|------|------|------------|
| `pipeline/parser.py` | 파일 형식별 텍스트 추출 (LlamaIndex SimpleDirectoryReader/FlatReader 래핑) | unstructured, PyMuPDF, python-docx, llama-index-core |
| `pipeline/cleaner.py` | 정규식 기반 텍스트 정제 | re, unicodedata |

### Layer 2: 처리 및 저장 (Processing & Storage)

| 파일 | 책임 | 주요 의존성 |
|------|------|------------|
| `pipeline/chunker.py` | 의미 단위 청킹, 오버랩 설정 (LlamaIndex SentenceSplitter 대안 지원) | langchain TextSplitter, llama-index-core |
| `pipeline/embedder.py` | 벡터 임베딩 생성 (배치 처리) | sentence-transformers, llama-index-embeddings-huggingface |
| `pipeline/vectorstore.py` | ChromaDB CRUD, 컬렉션 관리 (LlamaIndex ChromaVectorStore 통합) | chromadb, llama-index-vector-stores-chroma |

### Layer 3: 검색 및 생성 (Retrieval & Generation)

| 파일 | 책임 | 주요 의존성 |
|------|------|------------|
| `pipeline/retriever.py` | 하이브리드 검색 (Vector + BM25 + RRF, LlamaIndex VectorIndexRetriever 래핑) | chromadb, rank_bm25, llama-index-core |
| `pipeline/reranker.py` | Cross-encoder 재순위 | sentence-transformers |
| `pipeline/generator.py` | 프롬프트 구성 + LLM 호출 | langchain, ollama/openai |

### Layer 4: 인터페이스 (Interface)

| 파일 | 책임 | 주요 의존성 |
|------|------|------------|
| `api/ingest.py` | 문서 업로드 엔드포인트 | FastAPI, pipeline modules |
| `api/query.py` | 질의 응답 엔드포인트 | FastAPI, pipeline modules |
| `models/request.py` | 요청 Pydantic 스키마 | Pydantic v2 |
| `models/response.py` | 응답 Pydantic 스키마 | Pydantic v2 |
| `main.py` | FastAPI 앱 진입점 | FastAPI, Uvicorn |
| `app.py` | Gradio 데모 UI 진입점 | Gradio |
| `config/settings.py` | 전역 설정 (pydantic-settings) | pydantic-settings, python-dotenv |

### Layer 5: 오케스트레이션 (Orchestration)

| 파일 | 책임 | 주요 의존성 |
|------|------|------------|
| `pipeline/graphs/state.py` | 그래프 상태 정의 (TypedDict) | langgraph |
| `pipeline/graphs/ingest_graph.py` | LangGraph 인덱싱 워크플로우 StateGraph | langgraph |
| `pipeline/graphs/query_graph.py` | LangGraph 질의 응답 워크플로우 StateGraph | langgraph |

### Layer 6: 평가 (Evaluation)

| 파일 | 책임 | 주요 의존성 |
|------|------|------------|
| `pipeline/evaluator.py` | RAG 평가 메트릭 계산 (RAGAS) | ragas |
| `api/evaluate.py` | 평가 API 엔드포인트 | FastAPI, pipeline/evaluator.py |
| `models/evaluation.py` | 평가 요청/응답 Pydantic 스키마 | Pydantic v2 |

---

## 기술 선택 및 버전 핀닝

### 핵심 의존성 (tech.md 기반)

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

# 임베딩 및 ML
sentence-transformers>=3.3.0
torch>=2.5.0

# Vector DB
chromadb>=0.6.0

# 하이브리드 검색
rank-bm25>=0.2.2

# LangChain
langchain>=0.3.0
langchain-community>=0.3.0

# LangGraph
langgraph>=0.2.0
langchain-core>=0.3.0

# LlamaIndex
llama-index-core>=0.11.0
llama-index-vector-stores-chroma>=0.1.0
llama-index-embeddings-huggingface>=0.3.0

# RAG 평가
ragas>=0.1.0
datasets>=2.0.0  # RAGAS dependency

# 설정
python-dotenv>=1.0.0
pydantic>=2.10.0
pydantic-settings>=2.6.0

# 테스트
pytest>=8.3.0
pytest-asyncio>=0.24.0
httpx>=0.28.0
```

### 의존성 설치 순서

1. **기본 환경**: `python -m venv venv && venv\Scripts\activate`
2. **PyTorch 우선 설치** (GPU 사용 시 CUDA 버전에 맞춰 설치)
3. **sentence-transformers** (PyTorch 의존)
4. **chromadb** (sqlite3 빌드 의존성 주의)
5. **LangChain 및 기타 라이브러리**
6. **테스트 의존성**

---

## 구현 단계 (Priority 기반 Milestone)

### Primary Goal: 핵심 파이프라인 구축

**우선순위: High**

구현 순서 (TDD Red-Green-Refactor):

1. **config/settings.py** - 전역 설정 모듈 (pydantic-settings 기반)
   - 환경 변수 로딩, 기본값 정의
   - 테스트: 설정 로딩 검증

2. **pipeline/parser.py** - 문서 파싱 모듈
   - PDF, DOCX, TXT, MD, HTML 파서 구현
   - 테스트: 각 형식별 파싱 검증, 미지원 형식 거부 검증

3. **pipeline/cleaner.py** - 텍스트 정제 모듈
   - 정규식 기반 정제, 인코딩 처리
   - 테스트: 정제 규칙별 단위 테스트

4. **pipeline/chunker.py** - Chunk 분할 모듈
   - RecursiveCharacterTextSplitter 래핑, 메타데이터 부착
   - 테스트: 청크 크기, 오버랩, 메타데이터 검증

5. **pipeline/embedder.py** - 임베딩 생성 모듈
   - BAAI/bge-m3 모델 로딩, 배치 임베딩
   - 테스트: 벡터 차원 검증, 배치 처리 검증

6. **pipeline/vectorstore.py** - Vector DB 저장/조회 모듈
   - ChromaDB PersistentClient 래핑, CRUD 구현
   - 테스트: 저장/조회/삭제 검증

### Secondary Goal: 오케스트레이션 및 검색 파이프라인

**우선순위: High**

7. **pipeline/graphs/state.py** - 그래프 상태 정의
   - Ingest/Query 그래프 TypedDict 상태 모델 정의
   - 테스트: 상태 모델 타입 검증

8. **pipeline/graphs/ingest_graph.py** - LangGraph 인덱싱 그래프
   - parse→clean→chunk→embed→store 노드 정의 및 StateGraph 구성
   - 테스트: 그래프 노드 실행 순서 검증, 상태 전달 검증

9. **pipeline/graphs/query_graph.py** - LangGraph 질의 그래프
   - retrieve→rerank→generate 노드 정의, 조건부 라우팅(빈 컬렉션)
   - 테스트: 그래프 노드 실행 검증, 조건부 라우팅 검증

10. **pipeline/retriever.py** - 하이브리드 검색 모듈
    - Vector 검색 + BM25 검색 + RRF 병합
    - 테스트: 각 검색 방식 개별 검증, RRF 병합 로직 검증

11. **pipeline/reranker.py** - Reranker 모듈
    - Cross-encoder 모델 로딩, 재순위 로직
    - 테스트: 재순위 결과 검증

12. **pipeline/generator.py** - LLM 응답 생성 모듈
    - 프롬프트 템플릿, Ollama/OpenAI Provider 전환
    - 테스트: Provider 전환 검증, 타임아웃 검증

### Tertiary Goal: API, UI 및 평가 인터페이스

**우선순위: Medium**

13. **models/request.py, models/response.py** - Pydantic 스키마
    - 요청/응답 데이터 모델 정의
    - 테스트: 스키마 검증

14. **models/evaluation.py** - 평가 Pydantic 스키마
    - 평가 요청/응답 데이터 모델 정의
    - 테스트: 평가 스키마 검증

15. **pipeline/evaluator.py** - RAG 평가 모듈
    - RAGAS 기반 faithfulness, answer_relevancy, context_precision, context_recall 메트릭 계산
    - 테스트: 메트릭 계산 검증 (mock LLM 사용)

16. **api/ingest.py** - 문서 업로드 API
    - `/ingest/file`, `/ingest/folder` 엔드포인트
    - 테스트: 파일 업로드 통합 테스트

17. **api/query.py** - 질의 응답 API
    - `/query`, `/collections`, `/health` 엔드포인트
    - 테스트: 질의 응답 통합 테스트

18. **api/evaluate.py** - 평가 API
    - `POST /evaluate`, `POST /evaluate/batch` 엔드포인트
    - 테스트: 평가 API 통합 테스트

19. **main.py** - FastAPI 앱 조립
    - 라우터 등록, CORS, 예외 핸들러
    - 테스트: 앱 시작 및 라우팅 검증

20. **app.py** - Gradio Demo UI
    - 파일 업로드 + 채팅 인터페이스
    - 테스트: UI 컴포넌트 렌더링 검증

### Optional Goal: 성능 최적화 및 부가 기능

**우선순위: Low**

- 스트리밍 응답 (SSE) 구현
- GPU 가속 자동 감지 및 활용
- 임베딩/Reranker 모델 캐싱 최적화
- 비동기 파이프라인 실행 (대용량 파일)
- 다중 컬렉션 관리 UI

---

## 리스크 분석 및 대응

### Risk 1: 모델 다운로드 시간

- **설명**: BAAI/bge-m3 (~2GB), cross-encoder (~100MB) 최초 다운로드 소요
- **영향**: 최초 실행 시 긴 대기 시간
- **대응**: 진행률 표시, 사전 다운로드 스크립트 제공, 모델 캐시 경로 설정

### Risk 2: 메모리 사용량

- **설명**: 임베딩 모델 + Reranker 모델 동시 로딩 시 4~6GB RAM 사용
- **영향**: 8GB RAM 시스템에서 메모리 부족 가능
- **대응**: Lazy loading, 배치 크기 동적 조절, 모델별 순차 로딩 옵션

### Risk 3: Windows 경로 처리

- **설명**: Windows 역슬래시 경로, 긴 경로명, 한글 경로 문제
- **영향**: 파일 저장/로딩 실패
- **대응**: `pathlib.Path` 일관 사용, UTF-8 인코딩 명시, 긴 경로 지원 설정

### Risk 4: ChromaDB 버전 호환성

- **설명**: ChromaDB 0.6+에서 API 변경 가능성
- **영향**: 기존 코드 호환 실패
- **대응**: ChromaDB 래퍼 모듈(`vectorstore.py`)에서 API 추상화, 버전 고정

### Risk 5: Ollama 연결 실패

- **설명**: Ollama 미설치 또는 모델 미다운로드 상태에서 실행
- **영향**: LLM 응답 생성 불가
- **대응**: Health check에서 Ollama 상태 확인, OpenAI fallback 안내, 명확한 에러 메시지

### Risk 6: LlamaIndex/LangGraph 버전 호환성

- **설명**: LlamaIndex와 LangGraph는 빠른 업데이트 주기로 API 변경이 잦음
- **영향**: 버전 업그레이드 시 기존 코드 호환 실패 가능
- **대응**: 버전 핀닝 적용, 래퍼 모듈에서 API 추상화, 통합 테스트로 호환성 지속 검증

### Risk 7: RAGAS LLM 의존성

- **설명**: RAGAS 평가 시 별도 LLM 호출이 발생하여 API 비용 및 처리 시간 증가
- **영향**: 배치 평가 시 높은 API 비용, 평가 속도 저하
- **대응**: 평가용 LLM을 별도로 설정 가능하게 구현, 캐싱 적용, 배치 크기 제한

### Risk 8: LangGraph 상태 관리 복잡도

- **설명**: StateGraph TypedDict 설계가 부적절할 경우 노드 간 데이터 전달 오류 발생
- **영향**: 파이프라인 중간 단계에서 상태 손실 또는 타입 오류
- **대응**: TypedDict 상태 모델 신중히 설계, 각 노드 입출력 검증 테스트 작성

---

## 테스트 전략 (TDD)

### 단위 테스트 (Unit Tests)

각 pipeline 모듈에 대해 독립적 단위 테스트 작성:

- **test_parser.py**: 형식별 파싱, 미지원 형식 거부, 빈 파일 처리
- **test_cleaner.py**: 정제 규칙별 검증, 인코딩 처리
- **test_chunker.py**: 청크 크기/오버랩 검증, 메타데이터 확인
- **test_embedder.py**: 벡터 차원 검증, 배치 처리 (모델 mock 사용)
- **test_vectorstore.py**: CRUD 검증 (임시 ChromaDB 사용)
- **test_retriever.py**: Vector/BM25/RRF 개별 검증
- **test_reranker.py**: 재순위 로직 검증 (모델 mock 사용)
- **test_generator.py**: 프롬프트 구성, Provider 전환, 타임아웃
- **test_ingest_graph.py**: LangGraph 인덱싱 그래프 노드 실행 검증, 상태 전달 검증
- **test_query_graph.py**: LangGraph 질의 그래프 조건부 라우팅 검증, 빈 컬렉션 라우팅 검증
- **test_evaluator.py**: RAGAS 메트릭 계산 검증 (mock LLM 사용), 배치 평가 검증

### 통합 테스트 (Integration Tests)

- **test_api.py**: 엔드포인트 통합 테스트 (httpx AsyncClient)
- 전체 파이프라인 End-to-End 테스트 (소량 테스트 문서 사용)

### 테스트 도구

- **pytest**: 테스트 프레임워크
- **pytest-asyncio**: 비동기 테스트 지원
- **httpx**: FastAPI 테스트 클라이언트
- **unittest.mock / pytest-mock**: ML 모델 mock 처리

### 커버리지 목표

- 전체: 85% 이상
- pipeline/ 모듈: 90% 이상
- api/ 모듈: 80% 이상

---

## 참조

- **product.md**: 프로젝트 요구사항 및 사용 시나리오
- **structure.md**: 디렉토리 구조 및 모듈 설계
- **tech.md**: 기술 스택 상세 및 버전 정보
- **SPEC-RAG-001/spec.md**: EARS 형식 요구사항 명세
- **SPEC-RAG-001/acceptance.md**: 인수 테스트 시나리오

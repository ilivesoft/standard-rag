---
id: SPEC-RAG-001
type: acceptance
version: "1.1.0"
created: "2026-03-22"
updated: "2026-03-22"
author: jkh
---

# SPEC-RAG-001: 인수 테스트 기준

## 테스트 시나리오

---

### Scenario 1: 단일 PDF 파일 업로드 및 인덱싱

**관련 요구사항**: REQ-RAG-001, REQ-RAG-002, REQ-RAG-010~013

```gherkin
Given 시스템이 정상 가동 중이고 ChromaDB가 초기화되어 있다
  And 텍스트가 포함된 유효한 PDF 파일(test_document.pdf)이 준비되어 있다
When 사용자가 POST /ingest/file 엔드포인트로 해당 PDF 파일을 업로드한다
Then 시스템은 200 OK 응답을 반환한다
  And 응답에 인덱싱된 Chunk 수가 포함되어 있다
  And 파일이 storage/uploads/ 디렉토리에 저장되어 있다
  And ChromaDB에 해당 파일의 Chunk가 벡터와 메타데이터와 함께 저장되어 있다
  And 각 Chunk의 메타데이터에 원본 파일명과 Chunk 인덱스가 포함되어 있다
```

---

### Scenario 2: 폴더 일괄 업로드

**관련 요구사항**: REQ-RAG-003

```gherkin
Given 시스템이 정상 가동 중이다
  And 3개의 파일(PDF 1개, DOCX 1개, TXT 1개)이 업로드 준비되어 있다
When 사용자가 POST /ingest/folder 엔드포인트로 3개 파일을 일괄 업로드한다
Then 시스템은 200 OK 응답을 반환한다
  And 응답에 전체 처리 결과가 포함되어 있다 (성공: 3, 실패: 0)
  And 각 파일별 인덱싱된 Chunk 수가 응답에 포함되어 있다
  And ChromaDB에 모든 파일의 Chunk가 저장되어 있다
```

---

### Scenario 3: Vector 전용 검색 (alpha=1.0)

**관련 요구사항**: REQ-RAG-020, REQ-RAG-022

```gherkin
Given 시스템에 기술 문서가 인덱싱되어 있다
  And ChromaDB에 최소 10개 이상의 Chunk가 저장되어 있다
When 사용자가 POST /query 엔드포인트로 다음 질의를 전송한다:
  | 필드 | 값 |
  | query | "시스템의 전원을 끄는 방법은?" |
  | alpha | 1.0 |
  | top_k | 5 |
Then 시스템은 200 OK 응답을 반환한다
  And 검색 결과에 최대 5개의 문서 Chunk가 포함되어 있다
  And 각 결과에 텍스트, 점수, 메타데이터(파일명, Chunk 인덱스)가 포함되어 있다
  And 결과가 벡터 유사도 점수 기준 내림차순으로 정렬되어 있다
```

---

### Scenario 4: 하이브리드 검색 + Reranking 질의

**관련 요구사항**: REQ-RAG-020, REQ-RAG-021, REQ-RAG-030

```gherkin
Given 시스템에 다수의 문서가 인덱싱되어 있다
  And Ollama 또는 OpenAI LLM 서비스가 연결 가능하다
When 사용자가 POST /query 엔드포인트로 다음 질의를 전송한다:
  | 필드 | 값 |
  | query | "연간 매출 보고서의 핵심 요약은?" |
  | alpha | 0.5 |
  | top_k | 10 |
  | top_n | 3 |
Then 시스템은 200 OK 응답을 반환한다
  And 응답에 LLM이 생성한 자연어 답변이 포함되어 있다
  And 응답에 출처 정보(source documents)가 포함되어 있다
  And 출처 정보에 파일명과 관련 텍스트 스니펫이 포함되어 있다
  And 검색 결과는 Reranker에 의해 재순위된 상위 3개이다
```

---

### Scenario 5: LLM 응답 생성 (컨텍스트 기반)

**관련 요구사항**: REQ-RAG-030, REQ-RAG-031

```gherkin
Given 시스템에 제품 매뉴얼 문서가 인덱싱되어 있다
  And 환경 변수 LLM_PROVIDER가 "ollama"로 설정되어 있다
  And Ollama 서비스가 실행 중이며 llama3.2 모델이 사용 가능하다
When 사용자가 POST /query 엔드포인트로 "이 제품의 보증 기간은 얼마인가?" 질의를 전송한다
Then 시스템은 200 OK 응답을 반환한다
  And 응답의 answer 필드에 자연어 답변이 포함되어 있다
  And 답변 내용이 인덱싱된 문서의 관련 내용을 기반으로 한다
  And 응답의 sources 필드에 참조된 문서 출처가 포함되어 있다
  And 전체 응답 시간이 30초 이내이다
```

---

### Scenario 6: Edge Case - 빈 파일 및 미지원 형식 업로드

**관련 요구사항**: REQ-RAG-004, REQ-RAG-005

```gherkin
Given 시스템이 정상 가동 중이다

# 미지원 형식 거부
When 사용자가 POST /ingest/file 엔드포인트로 .exe 파일을 업로드한다
Then 시스템은 400 Bad Request 응답을 반환한다
  And 오류 메시지에 "지원하지 않는 파일 형식"이 포함되어 있다
  And 오류 메시지에 지원 형식 목록(PDF, DOCX, TXT, MD, HTML)이 포함되어 있다

# 빈 파일 거부
When 사용자가 POST /ingest/file 엔드포인트로 내용이 없는 빈 TXT 파일을 업로드한다
Then 시스템은 200 OK 응답을 반환한다
  And 응답에 해당 파일이 건너뛰기 처리되었음이 표시된다
  And 인덱싱된 Chunk 수는 0이다
```

---

### Scenario 7: 성능 게이트 - 질의 응답 시간

**관련 요구사항**: REQ-RAG-020, REQ-RAG-021, REQ-RAG-030

```gherkin
Given 시스템에 1000개 이상의 Chunk가 인덱싱되어 있다
  And 임베딩 모델과 Reranker 모델이 이미 로드되어 있다 (warm state)
  And LLM 서비스가 연결 가능하다
When 사용자가 POST /query 엔드포인트로 일반적인 질의를 전송한다
Then 전체 응답 시간(검색 + Reranking + LLM 생성)이 5초 이내이다
  And 검색 단계(Vector + BM25 + RRF)의 소요 시간이 1초 이내이다
  And Reranking 단계의 소요 시간이 1초 이내이다
```

---

### Scenario 8: 빈 컬렉션 검색 방지

**관련 요구사항**: REQ-RAG-023

```gherkin
Given 시스템이 정상 가동 중이다
  And ChromaDB에 인덱싱된 문서가 없다 (빈 컬렉션)
When 사용자가 POST /query 엔드포인트로 질의를 전송한다
Then 시스템은 200 OK 응답을 반환한다
  And 응답에 "인덱싱된 문서가 없습니다" 안내 메시지가 포함되어 있다
  And 검색 결과 목록이 비어 있다
  And LLM 호출은 실행되지 않는다
```

---

### Scenario 9: LLM 연결 실패 시 타임아웃 처리

**관련 요구사항**: REQ-RAG-032

```gherkin
Given 시스템에 문서가 인덱싱되어 있다
  And LLM 서비스(Ollama)가 실행 중이지 않다
When 사용자가 POST /query 엔드포인트로 질의를 전송한다
Then 시스템은 30초 이내에 503 Service Unavailable 응답을 반환한다
  And 오류 메시지에 "LLM 서비스에 연결할 수 없습니다"가 포함되어 있다
  And 검색 결과(Reranker까지의 결과)는 정상적으로 반환된다
```

---

### Scenario 10: Health Check 엔드포인트

**관련 요구사항**: REQ-RAG-043

```gherkin
Given 시스템이 정상 가동 중이다
When 사용자가 GET /health 엔드포인트로 요청한다
Then 시스템은 200 OK 응답을 반환한다
  And 응답에 다음 구성 요소의 상태가 포함되어 있다:
    | 구성 요소 | 상태 필드 |
    | ChromaDB | chromadb_status |
    | 임베딩 모델 | embedding_model_status |
    | LLM 서비스 | llm_status |
    | Reranker 모델 | reranker_status |
  And 각 구성 요소의 상태가 "healthy" 또는 "unhealthy"로 표시된다
```

---

### Scenario 11: LangGraph 인덱싱 그래프 실행

**관련 요구사항**: REQ-RAG-050, REQ-RAG-051

```gherkin
Given 시스템이 정상 가동 중이고 LangGraph Ingest Graph가 초기화되어 있다
  And 유효한 PDF 파일이 준비되어 있다
When 사용자가 POST /ingest/file 엔드포인트로 PDF 파일을 업로드한다
Then LangGraph Ingest Graph가 실행된다
  And parse → clean → chunk → embed → store 노드가 순서대로 실행된다
  And 각 노드의 실행 결과가 다음 노드의 상태(State)로 전달된다
  And 최종적으로 ChromaDB에 Chunk가 저장되고 200 OK 응답이 반환된다
```

---

### Scenario 12: LangGraph 조건부 라우팅 (빈 컬렉션)

**관련 요구사항**: REQ-RAG-053

```gherkin
Given 시스템이 정상 가동 중이다
  And ChromaDB에 인덱싱된 문서가 없다 (빈 컬렉션)
When 사용자가 POST /query 엔드포인트로 질의를 전송한다
Then LangGraph Query Graph가 실행된다
  And retrieve 노드에서 빈 결과가 감지된다
  And generate 노드는 실행되지 않고 조건부 라우팅으로 건너뛴다
  And "인덱싱된 문서가 없습니다" 안내 메시지가 반환된다
```

---

### Scenario 13: RAG 단일 평가 (Faithfulness + Answer Relevancy)

**관련 요구사항**: REQ-RAG-060, REQ-RAG-061

```gherkin
Given 시스템에 제품 매뉴얼이 인덱싱되어 있다
  And RAG 파이프라인으로 질의 응답이 완료된 결과가 있다:
    | question | "이 제품의 보증 기간은?" |
    | answer | "이 제품의 보증 기간은 1년입니다." |
    | contexts | ["제품 보증 기간은 구매일로부터 1년입니다.", "..."] |
When 사용자가 POST /evaluate 엔드포인트로 해당 결과를 제출한다
Then 시스템은 200 OK 응답을 반환한다
  And 응답에 faithfulness 점수가 포함되어 있다 (0.0 ~ 1.0)
  And 응답에 answer_relevancy 점수가 포함되어 있다 (0.0 ~ 1.0)
  And 응답에 context_precision 점수가 포함되어 있다 (0.0 ~ 1.0)
  And 각 메트릭 값이 0.0 이상 1.0 이하이다
```

---

### Scenario 14: RAG 배치 평가

**관련 요구사항**: REQ-RAG-062

```gherkin
Given 시스템이 정상 가동 중이다
  And 3개의 질문-답변-컨텍스트 쌍이 포함된 평가 데이터셋이 준비되어 있다
When 사용자가 POST /evaluate/batch 엔드포인트로 데이터셋을 제출한다
Then 시스템은 200 OK 응답을 반환한다
  And 응답에 데이터셋 전체의 평균 faithfulness 점수가 포함되어 있다
  And 응답에 데이터셋 전체의 평균 answer_relevancy 점수가 포함되어 있다
  And 응답에 3개 각각의 개별 평가 결과가 포함되어 있다
```

---

## Quality Gate 기준

### 기능 완성도

- [ ] 모든 Ubiquitous 요구사항 구현 완료
- [ ] 모든 Event-Driven 요구사항 구현 완료
- [ ] 모든 Unwanted 요구사항 구현 완료
- [ ] 모든 State-Driven 요구사항 구현 완료
- [ ] Optional 요구사항 구현 여부 확인
- [ ] LangGraph Ingest/Query Graph 노드 실행 검증
- [ ] RAG 평가 API 엔드포인트 (POST /evaluate, POST /evaluate/batch) 동작 확인

### 테스트 커버리지

- [ ] 단위 테스트 커버리지 85% 이상
- [ ] pipeline/ 모듈 테스트 커버리지 90% 이상
- [ ] 모든 14개 인수 테스트 시나리오 통과
- [ ] LangGraph 그래프 실행 테스트 통과
- [ ] RAGAS 메트릭 계산 테스트 통과

### 성능 기준

- [ ] 질의 응답 전체 시간 5초 이내 (warm state)
- [ ] 검색 단계 1초 이내
- [ ] Reranking 단계 1초 이내
- [ ] 단일 파일(10페이지 PDF) 인덱싱 30초 이내

### 코드 품질

- [ ] Ruff/Black 포맷팅 오류 없음
- [ ] 타입 힌트 100% 적용 (public API)
- [ ] 한국어 코드 주석 작성 (code_comments: ko)
- [ ] Pydantic v2 스키마 검증 적용

---

## Definition of Done

1. 모든 14개 인수 테스트 시나리오가 통과한다
2. `pytest tests/ -v --cov=pipeline --cov-report=term-missing` 실행 시 85% 이상 커버리지
3. FastAPI 서버가 포트 8000에서 정상 기동되며 `/docs`에서 Swagger 문서 확인 가능
4. Gradio UI가 포트 7860에서 정상 기동되며 파일 업로드 + 채팅 동작
5. `/health` 엔드포인트에서 모든 구성 요소 상태 확인 가능
6. Windows 10/11 환경에서 설치 및 실행 검증 완료
7. `.env.example` 파일에 모든 설정 변수가 문서화되어 있다
8. `POST /evaluate` 및 `POST /evaluate/batch` 엔드포인트 동작 확인
9. LangGraph Ingest/Query Graph 노드 실행 흐름 검증

---

## 검증 방법 및 도구

| 검증 항목 | 도구 | 명령어 |
|-----------|------|--------|
| 단위 테스트 | pytest | `pytest tests/ -v` |
| 커버리지 | pytest-cov | `pytest --cov=pipeline --cov-report=term-missing` |
| API 테스트 | httpx + pytest-asyncio | `pytest tests/test_api.py -v` |
| 수동 API 테스트 | Swagger UI | `http://localhost:8000/docs` |
| 수동 UI 테스트 | Gradio UI | `http://localhost:7860` |
| 코드 포맷팅 | Ruff | `ruff check . && ruff format --check .` |
| 타입 검사 | mypy (선택) | `mypy pipeline/ api/` |
| LangGraph 테스트 | pytest | `pytest tests/test_ingest_graph.py tests/test_query_graph.py -v` |
| 평가 테스트 | pytest | `pytest tests/test_evaluator.py -v` |

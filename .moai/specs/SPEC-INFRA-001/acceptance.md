# SPEC-INFRA-001 Acceptance Criteria

## Given/When/Then 시나리오

### Scenario 1: 개발 환경 기본값 — Chroma 자동 선택

**Given**: `.env` 파일에 `APP_ENV=development`만 설정되어 있고 `VECTORSTORE_BACKEND`는 미설정 상태
**When**: `python main.py`로 FastAPI 애플리케이션을 기동한다
**Then**:
- 기동 로그에 `"Vector backend: chroma"`가 출력된다
- `./storage/chroma` 디렉토리가 생성된다
- `/health` 엔드포인트가 `chromadb_status: "ok"`를 반환한다
- 기존 SPEC-RAG-001 기능(업로드, 질의, 삭제)이 모두 정상 동작한다

---

### Scenario 2: 운영 환경 기본값 — PGVector 자동 선택

**Given**:
- `docker compose -f docker-compose.dev.yml up -d`로 PostgreSQL이 기동되어 있음
- 환경 변수: `APP_ENV=production`, `PGVECTOR_PASSWORD=rag_dev_password`, `VECTORSTORE_BACKEND`는 미설정

**When**: `python main.py`로 애플리케이션을 기동한다
**Then**:
- 기동 로그에 `"Vector backend: pgvector"`가 출력된다
- PostgreSQL에 접속 성공 로그가 기록된다
- 파일 한 개 업로드 → `/query`로 질의 시 정상 응답이 반환된다
- `psql`로 `SELECT count(*) FROM langchain_pg_embedding` 실행 시 청크 수가 0보다 크다
- 동일 파일을 `/index/documents/{source}`로 삭제 후 다시 count하면 0이 된다

---

### Scenario 3: 명시적 백엔드 오버라이드

**Given**: `APP_ENV=production`이지만 `VECTORSTORE_BACKEND=chroma`로 명시 설정
**When**: 애플리케이션을 기동한다
**Then**:
- 선택된 백엔드는 `chroma`이다 (명시값이 자동 선택을 오버라이드)
- ChromaDB가 사용되고 PostgreSQL 연결 시도는 발생하지 않는다

---

### Scenario 4: 알 수 없는 백엔드 거부 (Fail-Fast)

**Given**: `VECTORSTORE_BACKEND=qdrant` (지원하지 않는 값)
**When**: 애플리케이션 기동 시 `create_vectorstore(settings)`가 호출된다
**Then**:
- `ValueError`가 발생한다
- 오류 메시지는 `"Unsupported VECTORSTORE_BACKEND: qdrant"`를 포함한다
- FastAPI 서버는 기동되지 **않는다**

---

### Scenario 5: PostgreSQL 연결 실패 시 Fail-Fast

**Given**: `APP_ENV=production`, PGVECTOR_HOST=`localhost`이지만 PostgreSQL이 **기동되지 않은** 상태
**When**: 애플리케이션을 기동한다
**Then**:
- 기동이 **즉시 실패**한다 (무한 대기 금지)
- 로그에 접속 정보(호스트, 포트, DB명)와 실패 원인이 포함된다
- 비밀번호는 로그에 노출되지 **않는다**

---

### Scenario 6: 임베딩 차원 불일치 검증

**Given**: `PGVECTOR_DIMENSION=768`로 설정되어 있으나 embedder는 1024 차원 벡터 생성
**When**: `/ingest/file`로 파일을 업로드한다 (`PGVectorStore.add()` 호출)
**Then**:
- `ValueError`가 발생한다
- 오류 메시지에 기대 차원(768)과 실제 차원(1024)이 포함된다

---

### Scenario 7: 소비자 인터페이스 호환성 (회귀)

**Given**: 기존 SPEC-RAG-001 구현 (retriever, graphs, index API)
**When**: `VectorStoreProtocol` 도입 후 Chroma 백엔드로 테스트 스위트 실행
**Then**:
- `tests/test_vectorstore.py` (기존 Chroma 테스트) 전부 통과
- `tests/test_retriever.py`, `tests/test_graphs*` 전부 통과
- 타입 힌트만 변경되고 런타임 동작은 동일함을 확인

---

### Scenario 8: PGVector 하이브리드 검색 호환성

**Given**: PGVector 백엔드에 tokenized_text 메타데이터를 포함한 청크 20개가 저장됨
**When**: `/query`로 질의를 전송한다 (하이브리드 검색 실행)
**Then**:
- `HybridRetriever`가 `get_all_texts_and_tokens()` 호출 시 20개 튜플을 반환
- 각 튜플의 두 번째 요소(tokens)는 리스트 형식으로 역직렬화됨
- BM25 인덱스가 정상 빌드되어 키워드 검색이 동작

---

### Scenario 9: 환경 전환 시나리오 (수동 재인덱싱)

**Given**: 개발 환경에서 Chroma로 문서 인덱싱 완료
**When**: `APP_ENV=production`으로 전환하여 PGVector로 기동
**Then**:
- PGVector에는 기존 데이터가 존재하지 **않는다** (별도 백엔드이므로 자동 마이그레이션 없음)
- 사용자가 `/ingest/file`로 수동 재인덱싱하면 정상 동작
- 마이그레이션 미지원은 SPEC-INFRA-001에 명시된 가정이며 오류가 아니다

---

## 엣지 케이스

### Edge 1: 빈 컬렉션 질의 (PGVector)

**Given**: PGVector 백엔드에 인덱싱된 문서가 0개
**When**: `/query`로 질의 전송
**Then**: `is_empty_collection=True` 처리되어 "문서가 없습니다" 안내 메시지 반환 (기존 SPEC-RAG-001 REQ-RAG-023 준수)

### Edge 2: 대소문자 APP_ENV

**Given**: `APP_ENV=Production` (대문자 포함)
**When**: `resolve_vectorstore_backend()` 호출
**Then**: 소문자로 정규화하여 `production`으로 처리 (관대한 파싱)

### Edge 3: docker-compose 미실행 상태에서 dev 기본 사용

**Given**: `APP_ENV=development`, `VECTORSTORE_BACKEND` 미설정, Docker 미설치
**When**: 애플리케이션 기동
**Then**: Chroma 백엔드가 자동 선택되어 Docker와 무관하게 정상 기동

### Edge 4: 동일 source 재업로드

**Given**: PGVector에 `file_a.pdf` 10청크 저장
**When**: 동일 `file_a.pdf`를 `/ingest/file`로 재업로드
**Then**: 기존 동작(덮어쓰기 또는 중복 방지)이 Chroma와 **동일한 방식**으로 처리됨 (SPEC-RAG-001 준수)

---

## 성능 및 품질 게이트

| 항목 | 목표 | 측정 방법 |
|------|------|-----------|
| 신규 모듈 테스트 커버리지 | 90%+ | `pytest --cov=pipeline.vectorstore_pgvector --cov=pipeline.vectorstore_factory --cov=pipeline.vectorstore_protocol` |
| 전체 프로젝트 커버리지 | 85%+ | `pytest --cov=pipeline --cov=api --cov=config` |
| 기존 테스트 회귀 | 0건 | CI에서 SPEC-RAG-001 테스트 156개 전부 통과 |
| mypy 타입 체크 | 0 오류 | `mypy pipeline/ api/ config/` |
| ruff 린트 | 0 오류 | `ruff check .` |
| PGVector `search()` 레이턴시 | < 200ms (top_k=10, 청크 1000개 기준) | 통합 테스트에서 측정 |
| 기동 시간 (Chroma) | < 5초 | 수동 계측 |
| 기동 시간 (PGVector, 로컬 Docker) | < 10초 | 수동 계측 |

---

## TRUST 5 검증 체크리스트

- [ ] **Tested**: 신규 모듈 90%+ 커버리지, 9개 인수 시나리오 전부 자동화 가능한 부분은 통합 테스트로 작성
- [ ] **Readable**: 모든 공개 메서드에 타입 힌트 + docstring, 한국어 주석 (code_comments: ko)
- [ ] **Unified**: pydantic-settings 기반 설정 구조 준수, 기존 네이밍 컨벤션 유지
- [ ] **Secured**: `PGVECTOR_PASSWORD`는 env-only, 로그 sanitize, 에러 메시지에 비밀정보 제외
- [ ] **Trackable**: 모든 커밋에 `SPEC-INFRA-001` 참조, conventional commits 형식

---

## 완료 조건

다음 모두 충족 시 SPEC-INFRA-001을 `completed`로 전환:

1. 위 9개 시나리오 중 자동화 가능한 부분(Scenario 1, 3, 4, 6, 7, 8, Edge 1, 2, 4)의 통합/단위 테스트 작성 및 통과
2. 수동 검증 시나리오(Scenario 2, 5, 9, Edge 3)는 체크리스트로 수행 결과 기록
3. 커버리지, mypy, ruff 품질 게이트 모두 통과
4. `.moai/project/tech.md`와 `README.md` 업데이트 완료
5. `docker-compose.dev.yml`로 실제 PGVector 기동 후 E2E 검증 1회 이상 수행

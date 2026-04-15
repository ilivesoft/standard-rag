# SPEC-INFRA-001 Implementation Plan

## 목표

ChromaDB 단일 백엔드를 `VectorStoreProtocol` 기반 다중 백엔드 아키텍처로 확장하고, PGVector(PostgreSQL) 신규 백엔드를 추가한다. 기존 소비자 코드(HybridRetriever, LangGraph 노드, API 라우터)는 인터페이스 호환성만으로 무변경 동작해야 한다.

---

## 태스크 분해

### Task 1: VectorStoreProtocol 정의 (Foundation)

**파일**: `pipeline/vectorstore_protocol.py` (신규)

- `typing.Protocol` 기반으로 10개 메서드 계약 정의
- 각 메서드에 docstring으로 의미/반환 형식 명시
- `runtime_checkable` 데코레이터 적용 (테스트 편의)

**검증**: 기존 `ChromaVectorStore`가 `isinstance(obj, VectorStoreProtocol)` 통과

---

### Task 2: 기존 Chroma 구현 네이밍 정리

**파일**: `pipeline/vectorstore.py` (수정)

- 클래스 `VectorStore` → `ChromaVectorStore`로 리네이밍
- 모듈 끝에 `VectorStore = ChromaVectorStore` alias 유지 (하위 호환)
- docstring 업데이트: "ChromaDB PersistentClient 기반 구현"
- 공개 메서드 시그니처 **변경 금지**

**검증**: `tests/test_vectorstore.py` 기존 테스트 전부 통과

---

### Task 3: settings.py 확장

**파일**: `config/settings.py` (수정)

- 신규 필드 9개 추가:
  - `APP_ENV: str = "development"`
  - `VECTORSTORE_BACKEND: str = ""`
  - `PGVECTOR_HOST`, `PGVECTOR_PORT`, `PGVECTOR_DATABASE`, `PGVECTOR_USER`, `PGVECTOR_PASSWORD`, `PGVECTOR_COLLECTION`, `PGVECTOR_DIMENSION`
- `resolve_vectorstore_backend() -> str` 메서드 추가 (자동 선택 로직)
  - `VECTORSTORE_BACKEND`가 빈 문자열이면 `APP_ENV`로 결정
  - 명시적 값이 있으면 그 값을 검증 후 반환
  - 알 수 없는 값이면 `ValueError`

**검증**: `tests/test_settings_env.py` 신규 작성 — 4가지 조합 검증

---

### Task 4: PGVectorStore 구현

**파일**: `pipeline/vectorstore_pgvector.py` (신규)

- 의존성: `langchain_postgres.PGVector`, `langchain_core.documents.Document`
- 생성자: `(host, port, database, user, password, collection, dimension)` + 연결 검증
- 10개 메서드 구현 (VectorStoreProtocol 전체):
  - `add`: `PGVector.add_embeddings()` 호출, 메타데이터는 JSONB로 저장
  - `search`: `PGVector.similarity_search_with_score_by_vector()` 사용, score 정규화
  - `count`: SQL `SELECT COUNT(*)` 직접 실행
  - `get_all_texts`, `get_all_texts_and_tokens`: SQL 전체 조회
  - `list_collections`: langchain-postgres 컬렉션 테이블 조회
  - `get_all_documents`, `get_by_source`: metadata->>'source' 필터
  - `delete_by_source`, `delete_all`: SQL DELETE

**핵심 주의사항**:
- Score 변환: langchain-postgres는 distance를 반환 → `score = 1 - distance`로 정규화
- Metadata JSONB: `tokenized_text`는 리스트이므로 JSON 직렬화 가능 확인
- Connection string: `postgresql+psycopg://user:pwd@host:port/db`

**검증**: `tests/test_vectorstore_pgvector.py` — docker-compose 기반 통합 테스트

---

### Task 5: 팩토리 구현

**파일**: `pipeline/vectorstore_factory.py` (신규)

```python
def create_vectorstore(settings) -> VectorStoreProtocol:
    backend = settings.resolve_vectorstore_backend()
    if backend == "chroma":
        return ChromaVectorStore(
            persist_dir=settings.CHROMA_PERSIST_DIR,
            collection=settings.CHROMA_COLLECTION,
        )
    if backend == "pgvector":
        return PGVectorStore(
            host=settings.PGVECTOR_HOST,
            port=settings.PGVECTOR_PORT,
            database=settings.PGVECTOR_DATABASE,
            user=settings.PGVECTOR_USER,
            password=settings.PGVECTOR_PASSWORD,
            collection=settings.PGVECTOR_COLLECTION,
            dimension=settings.PGVECTOR_DIMENSION,
        )
    raise ValueError(f"Unsupported VECTORSTORE_BACKEND: {backend}")
```

**검증**: `tests/test_vectorstore_factory.py` — 4가지 환경 조합 mocking

---

### Task 6: main.py DI 교체

**파일**: `main.py` (수정)

- `VectorStore(...)` 직접 생성 → `create_vectorstore(settings)` 호출로 교체
- import 정리: `from pipeline.vectorstore_factory import create_vectorstore`
- 타입 힌트 `VectorStoreProtocol`로 변경

**검증**: 기동 로그에서 선택된 백엔드 표시 확인

---

### Task 7: 소비자 타입 힌트 업데이트

**파일**: `pipeline/retriever.py`, `pipeline/graphs/*.py`, `api/index.py`, `api/query.py`

- 매개변수 타입을 `VectorStoreProtocol`로 변경
- 구체 클래스 import 제거
- **런타임 동작 변경 없음** (타입 힌트만 갱신)

**검증**: mypy 통과, 기존 테스트 통과

---

### Task 8: Docker Compose 및 초기화 스크립트

**파일**:
- `docker-compose.dev.yml` (신규)
- `docker/postgres-init.sql` (신규)

**docker-compose.dev.yml**:
```yaml
services:
  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_USER: rag_user
      POSTGRES_PASSWORD: rag_dev_password
      POSTGRES_DB: rag_vectors
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./docker/postgres-init.sql:/docker-entrypoint-initdb.d/init.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U rag_user -d rag_vectors"]
      interval: 5s
      timeout: 5s
      retries: 5
volumes:
  postgres_data:
```

**postgres-init.sql**:
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

**검증**: `docker compose -f docker-compose.dev.yml up -d` 후 `psql`로 확장 확인

---

### Task 9: 환경 템플릿 파일

**파일**: `.env.development`, `.env.production` (신규)

**.env.development**:
```
APP_ENV=development
VECTORSTORE_BACKEND=
# Chroma (기본)
CHROMA_PERSIST_DIR=./storage/chroma
CHROMA_COLLECTION=default

# PGVector (선택, docker-compose 사용 시)
# VECTORSTORE_BACKEND=pgvector
# PGVECTOR_HOST=localhost
# PGVECTOR_PASSWORD=rag_dev_password
```

**.env.production**:
```
APP_ENV=production
VECTORSTORE_BACKEND=
PGVECTOR_HOST=<SET_IN_ENV>
PGVECTOR_PORT=5432
PGVECTOR_DATABASE=<SET_IN_ENV>
PGVECTOR_USER=<SET_IN_ENV>
PGVECTOR_PASSWORD=<SET_IN_ENV>
PGVECTOR_COLLECTION=default
PGVECTOR_DIMENSION=1024
```

---

### Task 10: 의존성 업데이트

**파일**: `requirements.txt` (수정)

추가:
```
langchain-postgres>=0.0.12
psycopg[binary]>=3.2
sqlalchemy>=2.0
```

---

### Task 11: 문서 업데이트

**파일**: `.moai/project/tech.md`, `README.md` (수정)

- tech.md의 "Vector DB" 섹션에 PGVector 운영 옵션 추가
- README.md에 환경별 실행 가이드 추가 (docker-compose 사용법 포함)

---

## 빌드 순서 (의존성 기준)

1. **Task 1** (Protocol) → 인터페이스 정의
2. **Task 2** (Chroma 리네이밍) → 기존 구현이 Protocol 만족하는지 확인
3. **Task 3** (Settings 확장) → 팩토리가 사용할 설정 준비
4. **Task 4** (PGVectorStore) → docker-compose 없이는 통합 테스트 불가 → **Task 8과 병행**
5. **Task 8** (Docker Compose) → Task 4 통합 테스트 인프라
6. **Task 5** (Factory) → Task 1~4 완료 후
7. **Task 6** (main.py) → Task 5 완료 후
8. **Task 7** (소비자 타입 힌트) → 무순위, Task 1 후 언제든 가능
9. **Task 9** (.env 템플릿) → 무순위
10. **Task 10** (requirements.txt) → Task 4 시작 전
11. **Task 11** (문서) → 마지막

---

## 리스크 및 완화

| 리스크 | 영향 | 완화 방안 |
|--------|------|-----------|
| langchain-postgres API 불안정 | 구현 지연 | 버전 0.0.12 고정, 상위 버전 사전 검토 |
| pgvector 차원 불일치 런타임 오류 | 기동 실패 | 생성자에서 차원 검증, 명확한 에러 메시지 |
| BM25 tokenized_text JSONB 호환성 | 하이브리드 검색 실패 | 단위 테스트에서 직렬화/역직렬화 명시 검증 |
| Docker 미설치 개발자 | PGVector 테스트 불가 | Chroma를 기본값으로 유지, PGVector는 opt-in |
| 운영 DB 연결 정보 누설 | 보안 사고 | 로그에 비밀번호 미기록, 템플릿에 플레이스홀더만 |
| 기존 테스트 회귀 | SPEC-RAG-001 기능 파손 | Task 2에서 alias 유지, CI에서 기존 156개 테스트 검증 |

---

## 참조 구현

- **langchain-postgres 공식**: https://python.langchain.com/docs/integrations/vectorstores/pgvector/
- **pgvector 공식**: https://github.com/pgvector/pgvector
- **내부 참조**: `pipeline/vectorstore.py` (ChromaVectorStore) — 메서드 시그니처 모델

---

## MX 태그 계획

| 파일 | 태그 | 이유 |
|------|------|------|
| `pipeline/vectorstore_protocol.py` | `@MX:ANCHOR` | 모든 벡터 백엔드가 의존하는 계약 (fan_in >= 3) |
| `pipeline/vectorstore_factory.py` | `@MX:ANCHOR` | main.py가 유일한 DI 진입점 |
| `pipeline/vectorstore_pgvector.py` | `@MX:NOTE` | langchain-postgres 외부 의존 및 연결 풀 관리 |
| `pipeline/vectorstore_pgvector.py` | `@MX:WARN` | Fail-fast 연결 검증 (운영 영향) |

---

## TRUST 5 목표

| 항목 | 목표 |
|------|------|
| Tested | 85%+ 커버리지, 신규 모듈 90%+ |
| Readable | 타입 힌트, docstring, 한국어 주석 |
| Unified | 기존 pydantic-settings 패턴 준수 |
| Secured | 비밀번호 env-only, 로그 sanitize |
| Trackable | SPEC-INFRA-001 커밋 참조, conventional commits |

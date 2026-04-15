---
id: SPEC-INFRA-001
version: "1.0.0"
status: draft
created: "2026-04-15"
updated: "2026-04-15"
author: jkh
priority: high
issue_number: 0
---

## HISTORY

| 버전 | 날짜 | 작성자 | 변경 내용 |
|------|------|--------|-----------|
| 1.0.0 | 2026-04-15 | jkh | 최초 작성: 환경별 벡터 백엔드 분리 (Chroma dev / PGVector prod) |

---

# SPEC-INFRA-001: 환경별 벡터 백엔드 분리 (Chroma / PGVector)

## 개요

개발 환경(Development)과 운영 환경(Production)을 명확히 분리하고, 환경별로 서로 다른 벡터 데이터베이스 백엔드를 사용할 수 있도록 아키텍처를 확장한다. 개발 환경은 기존 ChromaDB(로컬 파일 기반)를 유지하여 설치 부담 없이 빠른 반복 개발이 가능하게 하고, 운영 환경은 PostgreSQL + pgvector 확장을 사용하여 트랜잭션, 백업, 멀티 클라이언트 동시성, 관리형 호스팅 등 엔터프라이즈 요구사항을 만족한다.

기존 `pipeline/vectorstore.py`의 ChromaDB 구체 구현을 `VectorStoreProtocol`로 추상화하고, 팩토리 패턴을 통해 `main.py` 기동 시점에 런타임 환경에 따른 구현체를 선택·주입한다. 기존 소비자(HybridRetriever, LangGraph Ingest/Query, Index API)는 인터페이스 호환성만으로 무변경 동작한다.

---

## Environment (환경)

- **공통 런타임**: Python 3.11+, FastAPI, LangChain, LangGraph (SPEC-RAG-001 기반)
- **개발 환경**:
  - 벡터 백엔드: ChromaDB PersistentClient (로컬 파일 기반, `./storage/chroma`)
  - 실행: `python main.py` 단독 (Docker 불필요)
  - 기본값: `APP_ENV=development`, `VECTORSTORE_BACKEND=chroma`
- **운영 환경**:
  - 벡터 백엔드: PostgreSQL 16 + pgvector extension (>= 0.7.0)
  - 실행: 별도 PostgreSQL 인스턴스 필요 (docker-compose 또는 관리형 DB)
  - 기본값: `APP_ENV=production`, `VECTORSTORE_BACKEND=pgvector`
- **로컬 개발용 PGVector 선택지**: `docker-compose.dev.yml`을 통해 `pgvector/pgvector:pg16` 이미지로 로컬 PostgreSQL 기동 가능

---

## Assumptions (가정)

- 개발자는 기본적으로 ChromaDB만으로 전체 파이프라인을 검증할 수 있다
- 운영 환경 PostgreSQL은 외부에서 프로비저닝되며, 애플리케이션은 접속 정보만 설정으로 받는다
- 임베딩 차원(`PGVECTOR_DIMENSION`, 기본 1024 for bge-m3)은 애플리케이션 설정으로 고정되며 런타임 변경은 지원하지 않는다
- 초기 버전은 두 백엔드 간 데이터 마이그레이션 도구를 제공하지 않는다 (필요 시 운영 환경에서 수동 재인덱싱)
- pgvector extension은 운영 DB에 이미 설치되어 있거나 `CREATE EXTENSION IF NOT EXISTS vector` 실행 권한이 있다
- 단일 애플리케이션 인스턴스 기준으로 설계하며, 다중 인스턴스 동시 쓰기는 PostgreSQL의 기본 동시성에 의존한다

---

## Requirements (요구사항)

### 모듈 1: 환경 설정 (Environment Configuration)

**[REQ-INFRA-001] Ubiquitous: 환경 변수 인식**

시스템은 **항상** 환경 변수 `APP_ENV`를 읽어 런타임 환경(`development` | `production`)을 결정해야 한다. 값이 지정되지 않으면 기본값 `development`를 사용한다.

**[REQ-INFRA-002] Ubiquitous: 벡터 백엔드 설정 필드**

`config/settings.py`는 **항상** 다음 필드를 제공해야 한다:
- `APP_ENV: str`
- `VECTORSTORE_BACKEND: str` (빈 문자열이면 환경별 자동 선택)
- `PGVECTOR_HOST`, `PGVECTOR_PORT`, `PGVECTOR_DATABASE`, `PGVECTOR_USER`, `PGVECTOR_PASSWORD`, `PGVECTOR_COLLECTION`, `PGVECTOR_DIMENSION`

**[REQ-INFRA-003] State-Driven: 환경별 기본 백엔드 자동 선택**

**IF** `VECTORSTORE_BACKEND`가 빈 문자열이고 `APP_ENV=production`이면 **THEN** 시스템은 `pgvector`를 선택해야 한다.
**IF** `VECTORSTORE_BACKEND`가 빈 문자열이고 `APP_ENV=development`이면 **THEN** 시스템은 `chroma`를 선택해야 한다.
**IF** `VECTORSTORE_BACKEND`가 명시적으로 설정되면 **THEN** 해당 값이 우선한다 (오버라이드).

**[REQ-INFRA-004] Unwanted: 알 수 없는 백엔드 거부**

`VECTORSTORE_BACKEND`가 `chroma`, `pgvector` 외의 값이면 시스템은 기동 **하지 않아야 한다**. 명확한 오류 메시지와 함께 즉시 실패해야 한다 (fail-fast).

---

### 모듈 2: VectorStore 추상화 (Abstraction)

**[REQ-INFRA-010] Ubiquitous: 프로토콜 정의**

시스템은 **항상** `pipeline/vectorstore_protocol.py`에 `VectorStoreProtocol`(`typing.Protocol`)을 정의해야 하며, 다음 메서드를 포함해야 한다:
- `add(chunks: list[dict], embeddings: list[list[float]]) -> int`
- `search(query_embedding: list[float], top_k: int = 10) -> list[dict]`
- `count() -> int`
- `get_all_texts() -> list[str]`
- `get_all_texts_and_tokens() -> tuple[list[str], list[list[str]]]`
- `list_collections() -> list[str]`
- `get_all_documents() -> list[dict]`
- `get_by_source(source: str) -> list[dict]`
- `delete_by_source(source: str) -> int`
- `delete_all() -> int`

**[REQ-INFRA-011] Unwanted: 기존 Chroma 인터페이스 파괴 금지**

기존 `pipeline/vectorstore.py`의 ChromaDB 구현 공개 메서드 시그니처는 **변경되어서는 안 된다**. 내부 네이밍 정리(`VectorStore` → `ChromaVectorStore`)는 허용되나, `VectorStore`는 하위 호환 alias로 유지해야 한다.

**[REQ-INFRA-012] Ubiquitous: 소비자의 프로토콜 의존**

`HybridRetriever`, LangGraph Ingest/Query 노드, Index API 라우터는 **항상** `VectorStoreProtocol` 타입 힌트를 사용해야 하며, 구체 클래스(`ChromaVectorStore`, `PGVectorStore`)를 직접 import 해서는 안 된다.

**[REQ-INFRA-013] Ubiquitous: 반환 스키마 통일**

모든 구현체의 `search()`는 **항상** `{text: str, metadata: dict, score: float}` 형식의 딕셔너리 리스트를 반환해야 하며, `score`는 `1 - cosine_distance` (범위 0~1, 높을수록 유사)로 정규화되어야 한다.

---

### 모듈 3: PGVector 구현 (Implementation)

**[REQ-INFRA-020] Ubiquitous: PGVectorStore 클래스**

시스템은 **항상** `pipeline/vectorstore_pgvector.py`에 `PGVectorStore` 클래스를 제공하며, `VectorStoreProtocol`을 완전히 구현해야 한다. 내부적으로 `langchain_postgres.PGVector`를 활용한다.

**[REQ-INFRA-021] Event-Driven: 사전 계산 임베딩 저장**

**WHEN** `PGVectorStore.add(chunks, embeddings)`가 호출되면 **THEN** 전달받은 벡터를 그대로 저장해야 하며, 내부에서 임베딩을 재계산해서는 안 된다.

**[REQ-INFRA-022] Ubiquitous: BM25 호환 메타데이터 유지**

`PGVectorStore`는 **항상** 청크 메타데이터(`source`, `chunk_index`, `tokenized_text` 등)를 JSONB 컬럼에 저장하고, `get_all_texts_and_tokens()` 호출 시 동일한 형식으로 반환해야 한다 (BM25 인덱스 빌드 호환).

**[REQ-INFRA-023] State-Driven: 연결 실패 시 Fail-Fast**

**IF** `PGVectorStore` 인스턴스 생성 또는 최초 커넥션 획득 시 PostgreSQL 연결이 실패하면 **THEN** 애플리케이션은 **즉시 기동 중단**하고 호스트/포트/DB 정보를 포함한 명확한 오류 메시지를 로깅해야 한다.

**[REQ-INFRA-024] State-Driven: 차원 불일치 검증**

**IF** `PGVECTOR_DIMENSION` 설정값과 실제 저장되는 임베딩 벡터 차원이 다르면 **THEN** 시스템은 `add()` 호출 시점에 ValueError를 발생시켜야 한다.

**[REQ-INFRA-025] Event-Driven: 소스 단위 삭제**

**WHEN** `delete_by_source(source)` 호출 시 **THEN** 해당 `source` 메타데이터를 가진 모든 행을 PostgreSQL에서 삭제하고 삭제 건수를 반환해야 한다. 존재하지 않는 source의 경우 0을 반환한다.

---

### 모듈 4: 팩토리 및 의존성 주입 (Factory & DI)

**[REQ-INFRA-030] Event-Driven: 팩토리 함수**

**WHEN** `main.py`가 애플리케이션 기동 시 **THEN** `pipeline/vectorstore_factory.py`의 `create_vectorstore(settings)` 함수를 호출하여 `VectorStoreProtocol` 인스턴스를 획득해야 한다.

**[REQ-INFRA-031] Ubiquitous: 단일 인스턴스 주입**

팩토리가 반환한 **단일 인스턴스**는 **항상** `HybridRetriever`, Ingest 그래프, Query 그래프, Index API 라우터에 주입되어야 한다. 컴포넌트별 별도 인스턴스 생성을 금지한다.

**[REQ-INFRA-032] Unwanted: main.py 외 팩토리 호출 금지**

`create_vectorstore()`는 `main.py`와 테스트 모듈 외에서 호출 **되어서는 안 된다**. API 라우터나 파이프라인 모듈이 팩토리를 직접 호출하면 DI 계층이 파괴된다.

---

### 모듈 5: 로컬 개발 환경 (Local Development)

**[REQ-INFRA-040] Ubiquitous: docker-compose 제공**

프로젝트는 **항상** 루트에 `docker-compose.dev.yml`을 제공하며, `pgvector/pgvector:pg16` 이미지를 사용하여 로컬 PostgreSQL 인스턴스를 기동할 수 있어야 한다.

**[REQ-INFRA-041] Event-Driven: 확장 자동 설치**

**WHEN** PostgreSQL 컨테이너가 최초 기동되면 **THEN** 초기화 스크립트(`docker/postgres-init.sql`)가 실행되어 `CREATE EXTENSION IF NOT EXISTS vector`를 수행해야 한다.

**[REQ-INFRA-042] Ubiquitous: 환경별 .env 템플릿**

프로젝트는 **항상** `.env.development`와 `.env.production` 템플릿 파일을 제공하여 각 환경의 설정 예시를 보여줘야 한다. 실제 비밀 정보는 포함하지 않는다.

---

## Specifications (명세)

### 기술 스택 (신규/추가)

| 구성 요소 | 기술 | 버전 | 역할 |
|-----------|------|------|------|
| 관계형 DB | PostgreSQL | 16 | 운영 벡터 저장소 |
| 벡터 확장 | pgvector | >= 0.7.0 | 벡터 타입 및 연산자 |
| LangChain 통합 | langchain-postgres | >= 0.0.12 | PGVector 래퍼 |
| DB 드라이버 | psycopg[binary] | >= 3.2 | PostgreSQL 드라이버 |
| ORM 의존성 | SQLAlchemy | >= 2.0 | langchain-postgres 내부 |
| 컨테이너 이미지 | pgvector/pgvector | pg16 | 로컬 개발용 |

### 설정 파라미터 (신규)

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| APP_ENV | development | 런타임 환경 (development / production) |
| VECTORSTORE_BACKEND | "" | 백엔드 선택 ("" = 자동, chroma, pgvector) |
| PGVECTOR_HOST | localhost | PostgreSQL 호스트 |
| PGVECTOR_PORT | 5432 | PostgreSQL 포트 |
| PGVECTOR_DATABASE | rag_vectors | DB 이름 |
| PGVECTOR_USER | rag_user | 접속 사용자 |
| PGVECTOR_PASSWORD | "" | 접속 비밀번호 (env로만 주입) |
| PGVECTOR_COLLECTION | default | 컬렉션/테이블명 |
| PGVECTOR_DIMENSION | 1024 | 임베딩 차원 (bge-m3) |

### 데이터 흐름

```
[기동 시점]
main.py → create_vectorstore(settings)
            ↓ VECTORSTORE_BACKEND 판독
            ├─ chroma → ChromaVectorStore(persist_dir, collection)
            └─ pgvector → PGVectorStore(host, port, db, user, pwd, collection, dim)
                              ↓ fail-fast 연결 검증
            ↓
            VectorStoreProtocol 인스턴스
            ↓ 단일 주입
            ┌──────────┬──────────┬──────────┐
            ↓          ↓          ↓          ↓
       HybridRetriever Ingest Graph Query Graph Index Router
```

---

## Traceability (추적성)

| 요구사항 ID | 모듈 | 구현 파일 | 테스트 파일 |
|-------------|------|-----------|-------------|
| REQ-INFRA-001~004 | 환경 설정 | config/settings.py | tests/test_settings_env.py |
| REQ-INFRA-010~013 | VectorStore 추상화 | pipeline/vectorstore_protocol.py, pipeline/vectorstore.py | tests/test_vectorstore_protocol.py |
| REQ-INFRA-020~025 | PGVector 구현 | pipeline/vectorstore_pgvector.py | tests/test_vectorstore_pgvector.py |
| REQ-INFRA-030~032 | 팩토리 및 DI | pipeline/vectorstore_factory.py, main.py | tests/test_vectorstore_factory.py |
| REQ-INFRA-040~042 | 로컬 개발 환경 | docker-compose.dev.yml, docker/postgres-init.sql, .env.development, .env.production | (수동 검증) |

---

## 의존성 및 관련 SPEC

- **SPEC-RAG-001** (완료): 본 SPEC은 SPEC-RAG-001이 구축한 파이프라인 구조(vectorstore, retriever, graphs, API)를 확장한다. 기존 ChromaDB 동작은 보존되어야 한다.

---

## 보안 고려사항

- `PGVECTOR_PASSWORD`는 코드/설정 파일에 커밋하지 않고 환경 변수로만 주입한다
- `.env.production` 템플릿은 실제 값 대신 플레이스홀더(`<SET_IN_ENV>`)만 포함한다
- PostgreSQL 연결은 가능한 경우 TLS를 사용해야 한다 (운영 환경 책임)
- 로그에는 비밀번호가 노출되지 않도록 유의한다 (호스트/포트/DB명까지만)

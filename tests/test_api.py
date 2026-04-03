# API 통합 테스트 - httpx.AsyncClient로 엔드포인트 검증
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

# TestClient를 사용하기 위해 main 앱을 import 전에 의존성 패치
import api.ingest as ingest_module
import api.query as query_module
import api.evaluate as evaluate_module


@pytest.fixture
def mock_ingest_graph():
    """인덱싱 그래프 모킹"""
    graph = MagicMock()
    graph.invoke.return_value = {
        "file_path": "/tmp/test.txt",
        "raw_text": "텍스트",
        "cleaned_text": "텍스트",
        "chunks": [{"text": "청크1", "metadata": {}}],
        "embeddings": [[0.1] * 384],
        "stored_count": 1,
        "error": None,
    }
    return graph


@pytest.fixture
def mock_query_graph():
    """질의 그래프 모킹"""
    graph = MagicMock()
    graph.invoke.return_value = {
        "query": "질문",
        "answer": "파이썬은 범용 언어입니다.",
        "sources": [{"source": "doc1.txt"}],
        "retrieved_chunks": [{"text": "청크1"}],
        "reranked_chunks": [{"text": "청크1"}],
        "error": None,
        "is_empty_collection": False,
    }
    return graph


@pytest.fixture
def mock_evaluator():
    """RAGEvaluator 모킹"""
    evaluator = MagicMock()
    evaluator.evaluate.return_value = {
        "faithfulness": 0.85,
        "answer_relevancy": 0.90,
        "context_precision": 0.75,
    }
    evaluator.batch_evaluate.return_value = {
        "average": {
            "faithfulness": 0.85,
            "answer_relevancy": 0.90,
            "context_precision": 0.75,
        },
        "results": [
            {
                "faithfulness": 0.85,
                "answer_relevancy": 0.90,
                "context_precision": 0.75,
            }
        ],
    }
    return evaluator


@pytest.fixture
def mock_vectorstore():
    """VectorStore 모킹"""
    vs = MagicMock()
    vs.count.return_value = 5
    vs.list_collections.return_value = ["default"]
    return vs


@pytest.fixture
def client(mock_ingest_graph, mock_query_graph, mock_evaluator, mock_vectorstore, tmp_path):
    """테스트용 FastAPI TestClient - 독립 앱으로 구성"""
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    # 의존성 주입
    ingest_module.set_ingest_graph(mock_ingest_graph)
    query_module.set_query_graph(mock_query_graph)
    query_module.set_dependencies(mock_vectorstore, MagicMock(is_loaded=False), MagicMock(is_loaded=False))
    evaluate_module.set_evaluator(mock_evaluator)

    # 업로드 디렉토리를 tmp_path로 설정
    upload_dir = str(tmp_path / "uploads")

    # 독립 테스트 앱 생성 (main.py 초기화 우회)
    test_app = FastAPI(title="Test RAG API")
    test_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    test_app.include_router(ingest_module.router)
    test_app.include_router(query_module.router)
    test_app.include_router(evaluate_module.router)

    @test_app.get("/")
    async def root():
        return {"name": "Test RAG Pipeline API", "version": "1.0.0"}

    # settings.UPLOAD_DIR를 tmp_path로 패치
    with patch("api.ingest.settings") as mock_settings:
        mock_settings.UPLOAD_DIR = upload_dir
        yield TestClient(test_app)


class TestRootEndpoint:
    """루트 엔드포인트 테스트"""

    def test_root_returns_200(self, client):
        """루트 엔드포인트가 200을 반환"""
        response = client.get("/")
        assert response.status_code == 200

    def test_root_has_name(self, client):
        """루트 응답에 name이 포함"""
        response = client.get("/")
        assert "name" in response.json()


class TestIngestFileEndpoint:
    """파일 인덱싱 엔드포인트 테스트"""

    def test_ingest_txt_file_returns_200(self, client):
        """TXT 파일 인덱싱이 200을 반환"""
        file_content = "test content".encode("utf-8")
        response = client.post(
            "/ingest/file",
            files={"file": ("test.txt", file_content, "text/plain")},
        )
        assert response.status_code == 200

    def test_ingest_unsupported_file_returns_400(self, client):
        """지원하지 않는 파일 형식은 400을 반환"""
        response = client.post(
            "/ingest/file",
            files={"file": ("test.exe", b"binary content", "application/octet-stream")},
        )
        assert response.status_code == 400

    def test_ingest_response_has_filename(self, client):
        """인덱싱 응답에 filename이 포함"""
        response = client.post(
            "/ingest/file",
            files={"file": ("test.txt", "content".encode("utf-8"), "text/plain")},
        )
        assert "filename" in response.json()

    def test_ingest_response_has_chunks_indexed(self, client):
        """인덱싱 응답에 chunks_indexed가 포함"""
        response = client.post(
            "/ingest/file",
            files={"file": ("test.txt", "content".encode("utf-8"), "text/plain")},
        )
        assert "chunks_indexed" in response.json()


class TestIngestFolderEndpoint:
    """폴더 인덱싱 엔드포인트 테스트"""

    def test_ingest_folder_returns_200(self, client):
        """다중 파일 인덱싱이 200을 반환"""
        response = client.post(
            "/ingest/folder",
            files=[
                ("files", ("a.txt", "content a".encode("utf-8"), "text/plain")),
                ("files", ("b.txt", "content b".encode("utf-8"), "text/plain")),
            ],
        )
        assert response.status_code == 200

    def test_ingest_folder_response_has_total(self, client):
        """폴더 응답에 total이 포함"""
        response = client.post(
            "/ingest/folder",
            files=[
                ("files", ("a.txt", "content".encode("utf-8"), "text/plain")),
            ],
        )
        data = response.json()
        assert "total" in data

    def test_ingest_folder_response_has_success(self, client):
        """폴더 응답에 success가 포함"""
        response = client.post(
            "/ingest/folder",
            files=[
                ("files", ("a.txt", "content".encode("utf-8"), "text/plain")),
            ],
        )
        data = response.json()
        assert "success" in data

    def test_ingest_folder_skips_unsupported(self, client):
        """폴더 인덱싱에서 미지원 파일은 건너뜀"""
        response = client.post(
            "/ingest/folder",
            files=[
                ("files", ("a.txt", "content".encode("utf-8"), "text/plain")),
                ("files", ("b.exe", "binary".encode("utf-8"), "application/octet-stream")),
            ],
        )
        data = response.json()
        assert data["total"] == 2
        assert data["failed"] >= 1


class TestQueryEndpoint:
    """질의 엔드포인트 테스트"""

    def test_query_returns_200(self, client):
        """질의 요청이 200을 반환"""
        response = client.post(
            "/query",
            json={"query": "파이썬이란?"},
        )
        assert response.status_code == 200

    def test_query_response_has_answer(self, client):
        """질의 응답에 answer가 포함"""
        response = client.post(
            "/query",
            json={"query": "파이썬이란?"},
        )
        assert "answer" in response.json()

    def test_query_response_has_sources(self, client):
        """질의 응답에 sources가 포함"""
        response = client.post(
            "/query",
            json={"query": "파이썬이란?"},
        )
        assert "sources" in response.json()

    def test_query_with_custom_params(self, client):
        """커스텀 파라미터로 질의 가능"""
        response = client.post(
            "/query",
            json={"query": "테스트", "top_k": 5, "top_n": 2, "alpha": 0.7},
        )
        assert response.status_code == 200


class TestCollectionsEndpoint:
    """컬렉션 목록 엔드포인트 테스트"""

    def test_collections_returns_200(self, client):
        """컬렉션 목록 요청이 200을 반환"""
        response = client.get("/collections")
        assert response.status_code == 200

    def test_collections_has_collections_key(self, client):
        """컬렉션 응답에 collections 키가 포함"""
        response = client.get("/collections")
        assert "collections" in response.json()


class TestHealthEndpoint:
    """헬스체크 엔드포인트 테스트"""

    def test_health_returns_200(self, client):
        """헬스체크 요청이 200을 반환"""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_has_status(self, client):
        """헬스체크 응답에 status가 포함"""
        response = client.get("/health")
        assert "status" in response.json()


class TestEvaluateEndpoint:
    """평가 엔드포인트 테스트"""

    def test_evaluate_returns_200(self, client):
        """단일 평가 요청이 200을 반환"""
        response = client.post(
            "/evaluate",
            json={
                "question": "파이썬이란?",
                "answer": "파이썬은 범용 언어입니다.",
                "contexts": ["파이썬은 인터프리터 언어입니다."],
            },
        )
        assert response.status_code == 200

    def test_evaluate_response_has_faithfulness(self, client):
        """평가 응답에 faithfulness가 포함"""
        response = client.post(
            "/evaluate",
            json={
                "question": "질문",
                "answer": "답변",
                "contexts": ["컨텍스트"],
            },
        )
        assert "faithfulness" in response.json()

    def test_batch_evaluate_returns_200(self, client):
        """배치 평가 요청이 200을 반환"""
        response = client.post(
            "/evaluate/batch",
            json={
                "items": [
                    {
                        "question": "질문1",
                        "answer": "답변1",
                        "contexts": ["컨텍스트1"],
                    }
                ]
            },
        )
        assert response.status_code == 200

    def test_batch_evaluate_has_average(self, client):
        """배치 평가 응답에 average가 포함"""
        response = client.post(
            "/evaluate/batch",
            json={
                "items": [
                    {
                        "question": "질문1",
                        "answer": "답변1",
                        "contexts": ["컨텍스트1"],
                    }
                ]
            },
        )
        assert "average" in response.json()

# 문서 업로드 API - /ingest/file, /ingest/folder 엔드포인트
import shutil
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from config.settings import settings
from models.response import IngestResponse, FolderIngestResponse
from pipeline.parser import SUPPORTED_EXTENSIONS
from pipeline.exceptions import UnsupportedFileTypeError, EmptyFileError

# 의존성은 main.py에서 주입
_ingest_graph = None


def get_ingest_graph():
    """인덱싱 그래프 인스턴스를 반환합니다."""
    return _ingest_graph


def set_ingest_graph(graph):
    """인덱싱 그래프 인스턴스를 설정합니다."""
    global _ingest_graph
    _ingest_graph = graph


router = APIRouter(prefix="/ingest", tags=["ingest"])


@router.post("/file", response_model=IngestResponse)
async def ingest_file(file: UploadFile = File(...)):
    """단일 파일을 업로드하고 인덱싱합니다.

    Args:
        file: 업로드할 문서 파일 (PDF, DOCX, TXT, MD, HTML 지원)

    Returns:
        IngestResponse: 인덱싱 결과

    Raises:
        400: 지원하지 않는 파일 형식
        500: 인덱싱 처리 오류
    """
    filename = file.filename or "unknown"
    ext = Path(filename).suffix.lower()

    # 지원 형식 검증
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"지원하지 않는 파일 형식입니다: {ext}. 지원 형식: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    # 업로드 디렉토리 생성
    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)

    # 파일 저장
    save_path = upload_dir / filename
    with save_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    # 인덱싱 그래프 실행
    graph = get_ingest_graph()
    if graph is None:
        raise HTTPException(status_code=500, detail="인덱싱 그래프가 초기화되지 않았습니다.")

    try:
        initial_state = {
            "file_path": str(save_path),
            "raw_text": "",
            "cleaned_text": "",
            "chunks": [],
            "embeddings": [],
            "stored_count": 0,
            "error": None,
        }
        result = graph.invoke(initial_state)

        if result.get("error"):
            return IngestResponse(
                filename=filename,
                chunks_indexed=0,
                skipped=True,
                skip_reason=result["error"],
            )

        return IngestResponse(
            filename=filename,
            chunks_indexed=result["stored_count"],
        )

    except (UnsupportedFileTypeError, EmptyFileError) as e:
        return IngestResponse(
            filename=filename,
            chunks_indexed=0,
            skipped=True,
            skip_reason=str(e),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/folder", response_model=FolderIngestResponse)
async def ingest_folder(files: list[UploadFile] = File(...)):
    """여러 파일을 업로드하고 인덱싱합니다.

    Args:
        files: 업로드할 문서 파일 리스트

    Returns:
        FolderIngestResponse: 폴더 인덱싱 결과
    """
    results = []
    success_count = 0
    failed_count = 0

    for file in files:
        try:
            # 파일별로 ingest_file 로직 재사용
            filename = file.filename or "unknown"
            ext = Path(filename).suffix.lower()

            if ext not in SUPPORTED_EXTENSIONS:
                results.append(IngestResponse(
                    filename=filename,
                    chunks_indexed=0,
                    skipped=True,
                    skip_reason=f"지원하지 않는 파일 형식: {ext}",
                ))
                failed_count += 1
                continue

            upload_dir = Path(settings.UPLOAD_DIR)
            upload_dir.mkdir(parents=True, exist_ok=True)
            save_path = upload_dir / filename

            with save_path.open("wb") as f:
                shutil.copyfileobj(file.file, f)

            graph = get_ingest_graph()
            if graph is None:
                results.append(IngestResponse(
                    filename=filename,
                    chunks_indexed=0,
                    skipped=True,
                    skip_reason="인덱싱 그래프 미초기화",
                ))
                failed_count += 1
                continue

            initial_state = {
                "file_path": str(save_path),
                "raw_text": "",
                "cleaned_text": "",
                "chunks": [],
                "embeddings": [],
                "stored_count": 0,
                "error": None,
            }
            result = graph.invoke(initial_state)

            if result.get("error"):
                results.append(IngestResponse(
                    filename=filename,
                    chunks_indexed=0,
                    skipped=True,
                    skip_reason=result["error"],
                ))
                failed_count += 1
            else:
                results.append(IngestResponse(
                    filename=filename,
                    chunks_indexed=result["stored_count"],
                ))
                success_count += 1

        except Exception as e:
            results.append(IngestResponse(
                filename=file.filename or "unknown",
                chunks_indexed=0,
                skipped=True,
                skip_reason=str(e),
            ))
            failed_count += 1

    return FolderIngestResponse(
        total=len(files),
        success=success_count,
        failed=failed_count,
        results=results,
    )

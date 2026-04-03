# OCR 파싱 테스트 - 이미지 파일 및 스캔 PDF OCR 폴백 검증
import sys
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

# easyocr 미설치 환경에서도 테스트 가능하도록 sys.modules에 mock 등록
_mock_easyocr_module = MagicMock()
sys.modules.setdefault("easyocr", _mock_easyocr_module)

from pipeline.parser import DocumentParser, SUPPORTED_EXTENSIONS  # noqa: E402
from pipeline.exceptions import EmptyFileError  # noqa: E402


@pytest.fixture
def parser():
    """DocumentParser 인스턴스"""
    return DocumentParser()


@pytest.fixture(autouse=True)
def reset_ocr_reader(parser):
    """각 테스트 전 OCR 리더를 초기화합니다."""
    parser._ocr_reader = None
    yield
    parser._ocr_reader = None


def _setup_mock_reader(readtext_return):
    """mock OCR 리더를 설정하고 sys.modules에 등록합니다."""
    mock_reader = MagicMock()
    mock_reader.readtext.return_value = readtext_return
    _mock_easyocr_module.Reader.return_value = mock_reader
    return mock_reader


class TestOcrSupportedExtensions:
    """이미지 확장자 지원 여부 테스트"""

    @pytest.mark.parametrize("ext", [".png", ".jpg", ".jpeg", ".bmp", ".tiff"])
    def test_image_extensions_supported(self, parser, ext):
        """이미지 확장자가 SUPPORTED_EXTENSIONS에 포함"""
        assert ext in SUPPORTED_EXTENSIONS
        assert parser.is_supported(Path(f"test{ext}")) is True


class TestParseImage:
    """이미지 파일 파싱 테스트"""

    def test_parse_image_returns_text(self, parser, tmp_path):
        """이미지 파일에서 OCR로 텍스트 추출"""
        img_file = tmp_path / "test.png"
        img_file.write_bytes(b"fake image data")

        mock_reader = _setup_mock_reader(["안녕하세요", "테스트 텍스트"])
        result = parser.parse(img_file)

        assert "안녕하세요" in result
        assert "테스트 텍스트" in result
        mock_reader.readtext.assert_called_once()

    @pytest.mark.parametrize("ext", [".jpg", ".jpeg", ".bmp", ".tiff"])
    def test_parse_various_image_formats(self, parser, tmp_path, ext):
        """다양한 이미지 형식 파싱"""
        img_file = tmp_path / f"test{ext}"
        img_file.write_bytes(b"fake image data")

        _setup_mock_reader(["추출된 텍스트"])
        result = parser.parse(img_file)

        assert "추출된 텍스트" in result

    def test_parse_image_empty_raises_error(self, parser, tmp_path):
        """OCR 결과가 비어있으면 EmptyFileError"""
        img_file = tmp_path / "empty.png"
        img_file.write_bytes(b"fake image data")

        _setup_mock_reader([])

        with pytest.raises(EmptyFileError):
            parser.parse(img_file)

    def test_ocr_reader_lazy_init(self, parser):
        """OCR 리더가 lazy 초기화됨"""
        assert parser._ocr_reader is None


class TestPdfOcrFallback:
    """PDF OCR 폴백 테스트"""

    def test_pdf_text_extraction_no_ocr(self, parser, tmp_path):
        """텍스트가 있는 PDF는 OCR을 사용하지 않음"""
        pdf_file = tmp_path / "text.pdf"
        pdf_file.write_bytes(b"dummy pdf")

        mock_page = MagicMock()
        mock_page.get_text.return_value = "텍스트 PDF 내용"
        mock_doc = MagicMock()
        mock_doc.__iter__ = lambda self: iter([mock_page])
        mock_doc.__enter__ = lambda self: mock_doc
        mock_doc.__exit__ = MagicMock(return_value=False)

        with patch("fitz.open", return_value=mock_doc):
            result = parser.parse(pdf_file)

        assert "텍스트 PDF 내용" in result
        mock_page.get_pixmap.assert_not_called()

    def test_pdf_ocr_fallback_on_empty_page(self, parser, tmp_path):
        """텍스트가 없는 PDF 페이지에서 OCR 폴백 사용"""
        pdf_file = tmp_path / "scanned.pdf"
        pdf_file.write_bytes(b"dummy pdf")

        mock_pixmap = MagicMock()
        mock_pixmap.tobytes.return_value = b"fake png bytes"

        mock_page = MagicMock()
        mock_page.get_text.return_value = "   "
        mock_page.get_pixmap.return_value = mock_pixmap

        mock_doc = MagicMock()
        mock_doc.__iter__ = lambda self: iter([mock_page])
        mock_doc.__enter__ = lambda self: mock_doc
        mock_doc.__exit__ = MagicMock(return_value=False)

        _setup_mock_reader(["OCR로 추출된 텍스트"])

        with patch("fitz.open", return_value=mock_doc):
            result = parser.parse(pdf_file)

        assert "OCR로 추출된 텍스트" in result
        mock_page.get_pixmap.assert_called_once()

    def test_pdf_mixed_pages(self, parser, tmp_path):
        """텍스트 페이지와 스캔 페이지가 혼합된 PDF"""
        pdf_file = tmp_path / "mixed.pdf"
        pdf_file.write_bytes(b"dummy pdf")

        text_page = MagicMock()
        text_page.get_text.return_value = "텍스트 페이지 내용"

        scan_pixmap = MagicMock()
        scan_pixmap.tobytes.return_value = b"fake png bytes"
        scan_page = MagicMock()
        scan_page.get_text.return_value = ""
        scan_page.get_pixmap.return_value = scan_pixmap

        mock_doc = MagicMock()
        mock_doc.__iter__ = lambda self: iter([text_page, scan_page])
        mock_doc.__enter__ = lambda self: mock_doc
        mock_doc.__exit__ = MagicMock(return_value=False)

        _setup_mock_reader(["스캔 페이지 OCR 결과"])

        with patch("fitz.open", return_value=mock_doc):
            result = parser.parse(pdf_file)

        assert "텍스트 페이지 내용" in result
        assert "스캔 페이지 OCR 결과" in result

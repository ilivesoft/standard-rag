# 파서 모듈 테스트 - PDF, DOCX, TXT, MD, HTML 형식 파싱 검증
import pytest
from pathlib import Path
from pipeline.parser import DocumentParser
from pipeline.exceptions import UnsupportedFileTypeError, EmptyFileError


@pytest.fixture
def parser():
    """DocumentParser 인스턴스"""
    return DocumentParser()


class TestDocumentParserTxt:
    """TXT 파일 파싱 테스트"""

    def test_parse_txt_returns_text(self, parser, tmp_path):
        """TXT 파일에서 텍스트를 성공적으로 추출"""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("안녕하세요. 테스트 텍스트입니다.", encoding="utf-8")
        result = parser.parse(txt_file)
        assert "안녕하세요" in result
        assert "테스트 텍스트입니다" in result

    def test_parse_txt_multiline(self, parser, tmp_path):
        """여러 줄 TXT 파일 파싱"""
        txt_file = tmp_path / "multi.txt"
        txt_file.write_text("첫번째 줄\n두번째 줄\n세번째 줄", encoding="utf-8")
        result = parser.parse(txt_file)
        assert "첫번째 줄" in result
        assert "두번째 줄" in result


class TestDocumentParserMarkdown:
    """마크다운 파일 파싱 테스트"""

    def test_parse_md_returns_text(self, parser, tmp_path):
        """MD 파일에서 텍스트를 성공적으로 추출"""
        md_file = tmp_path / "test.md"
        md_file.write_text("# 제목\n\n내용입니다.", encoding="utf-8")
        result = parser.parse(md_file)
        assert "제목" in result
        assert "내용입니다" in result


class TestDocumentParserHtml:
    """HTML 파일 파싱 테스트"""

    def test_parse_html_removes_tags(self, parser, tmp_path):
        """HTML 태그를 제거하고 텍스트만 추출"""
        html_file = tmp_path / "test.html"
        html_file.write_text(
            "<html><body><h1>제목</h1><p>내용입니다.</p></body></html>",
            encoding="utf-8"
        )
        result = parser.parse(html_file)
        assert "제목" in result
        assert "내용입니다" in result
        assert "<html>" not in result
        assert "<h1>" not in result

    def test_parse_html_nested_tags(self, parser, tmp_path):
        """중첩 HTML 태그 제거"""
        html_file = tmp_path / "nested.html"
        html_file.write_text(
            "<div><span><b>굵은 텍스트</b></span></div>",
            encoding="utf-8"
        )
        result = parser.parse(html_file)
        assert "굵은 텍스트" in result
        assert "<div>" not in result


class TestDocumentParserPdf:
    """PDF 파일 파싱 테스트 (실제 파일이 없으므로 모킹 사용)"""

    def test_parse_pdf_calls_pymupdf(self, parser, tmp_path, monkeypatch):
        """PDF 파싱이 PyMuPDF를 호출함을 검증"""
        from unittest.mock import MagicMock, patch

        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"dummy pdf content")

        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_page.get_text.return_value = "PDF 텍스트 내용입니다."
        mock_doc.__iter__ = lambda self: iter([mock_page])
        mock_doc.__enter__ = lambda self: mock_doc
        mock_doc.__exit__ = MagicMock(return_value=False)

        with patch("fitz.open", return_value=mock_doc):
            result = parser.parse(pdf_file)
        assert "PDF 텍스트 내용입니다." in result


class TestDocumentParserDocx:
    """DOCX 파일 파싱 테스트 (모킹 사용)"""

    def test_parse_docx_returns_text(self, parser, tmp_path, monkeypatch):
        """DOCX 파일에서 텍스트 추출"""
        from unittest.mock import MagicMock, patch

        docx_file = tmp_path / "test.docx"
        docx_file.write_bytes(b"dummy docx content")

        mock_doc = MagicMock()
        mock_para1 = MagicMock()
        mock_para1.text = "DOCX 첫 번째 단락"
        mock_para2 = MagicMock()
        mock_para2.text = "DOCX 두 번째 단락"
        mock_doc.paragraphs = [mock_para1, mock_para2]
        mock_doc.tables = []

        with patch("docx.Document", return_value=mock_doc):
            result = parser.parse(docx_file)
        assert "DOCX 첫 번째 단락" in result
        assert "DOCX 두 번째 단락" in result


class TestDocumentParserErrors:
    """파서 오류 처리 테스트"""

    def test_unsupported_file_raises_error(self, parser, tmp_path):
        """.exe 등 지원하지 않는 파일 형식은 UnsupportedFileTypeError"""
        exe_file = tmp_path / "malware.exe"
        exe_file.write_bytes(b"not a document")
        with pytest.raises(UnsupportedFileTypeError):
            parser.parse(exe_file)

    def test_empty_txt_raises_error(self, parser, tmp_path):
        """빈 TXT 파일은 EmptyFileError"""
        empty_file = tmp_path / "empty.txt"
        empty_file.write_text("", encoding="utf-8")
        with pytest.raises(EmptyFileError):
            parser.parse(empty_file)

    def test_whitespace_only_raises_error(self, parser, tmp_path):
        """공백만 있는 파일은 EmptyFileError"""
        ws_file = tmp_path / "whitespace.txt"
        ws_file.write_text("   \n\t\n  ", encoding="utf-8")
        with pytest.raises(EmptyFileError):
            parser.parse(ws_file)

    def test_file_not_found_raises_error(self, parser, tmp_path):
        """존재하지 않는 파일은 FileNotFoundError"""
        missing = tmp_path / "missing.txt"
        with pytest.raises(FileNotFoundError):
            parser.parse(missing)


class TestUnstructuredParser:
    """UnstructuredFileLoader 기반 파싱 테스트"""

    def test_parse_with_unstructured_returns_text(self, parser, tmp_path):
        """_parse_with_unstructured가 텍스트를 반환"""
        from unittest.mock import MagicMock, patch
        from langchain_core.documents import Document

        txt_file = tmp_path / "test.txt"
        txt_file.write_text("언스트럭처드 텍스트", encoding="utf-8")

        mock_doc = Document(page_content="언스트럭처드 텍스트")
        mock_loader = MagicMock()
        mock_loader.load.return_value = [mock_doc]

        with patch("langchain_community.document_loaders.UnstructuredFileLoader", return_value=mock_loader):
            result = parser._parse_with_unstructured(txt_file)

        assert "언스트럭처드 텍스트" in result

    def test_parse_with_unstructured_joins_multiple_docs(self, parser, tmp_path):
        """여러 Document 반환 시 개행으로 결합"""
        from unittest.mock import MagicMock, patch
        from langchain_core.documents import Document

        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"dummy")

        docs = [
            Document(page_content="첫 번째 섹션"),
            Document(page_content="두 번째 섹션"),
        ]
        mock_loader = MagicMock()
        mock_loader.load.return_value = docs

        with patch("langchain_community.document_loaders.UnstructuredFileLoader", return_value=mock_loader):
            result = parser._parse_with_unstructured(pdf_file)

        assert "첫 번째 섹션" in result
        assert "두 번째 섹션" in result

    def test_parse_with_unstructured_skips_empty_docs(self, parser, tmp_path):
        """빈 page_content를 가진 Document는 결과에서 제외"""
        from unittest.mock import MagicMock, patch
        from langchain_core.documents import Document

        txt_file = tmp_path / "test.txt"
        txt_file.write_text("내용", encoding="utf-8")

        docs = [
            Document(page_content="유효한 내용"),
            Document(page_content="   "),
            Document(page_content=""),
        ]
        mock_loader = MagicMock()
        mock_loader.load.return_value = docs

        with patch("langchain_community.document_loaders.UnstructuredFileLoader", return_value=mock_loader):
            result = parser._parse_with_unstructured(txt_file)

        assert result == "유효한 내용"

    def test_parse_falls_back_to_unstructured_on_empty_primary(self, parser, tmp_path):
        """기본 파서가 빈 결과를 반환할 때 Unstructured 폴백 동작"""
        from unittest.mock import MagicMock, patch
        from langchain_core.documents import Document

        docx_file = tmp_path / "test.docx"
        docx_file.write_bytes(b"dummy")

        # 기본 docx 파서가 빈 결과를 반환하도록 설정
        mock_doc = MagicMock()
        mock_doc.paragraphs = []
        mock_doc.tables = []

        fallback_doc = Document(page_content="언스트럭처드로 추출된 내용")
        mock_loader = MagicMock()
        mock_loader.load.return_value = [fallback_doc]

        with patch("docx.Document", return_value=mock_doc):
            with patch("langchain_community.document_loaders.UnstructuredFileLoader", return_value=mock_loader):
                result = parser.parse(docx_file)

        assert "언스트럭처드로 추출된 내용" in result

    def test_parse_with_unstructured_mode_is_single(self, parser, tmp_path):
        """UnstructuredFileLoader가 mode='single'로 초기화됨을 검증"""
        from unittest.mock import MagicMock, patch
        from langchain_core.documents import Document

        txt_file = tmp_path / "test.txt"
        txt_file.write_text("내용", encoding="utf-8")

        mock_loader = MagicMock()
        mock_loader.load.return_value = [Document(page_content="내용")]

        with patch("langchain_community.document_loaders.UnstructuredFileLoader", return_value=mock_loader) as mock_cls:
            parser._parse_with_unstructured(txt_file)

        mock_cls.assert_called_once_with(str(txt_file), mode="single")


class TestDocumentParserIsSupported:
    """지원 형식 확인 테스트"""

    def test_txt_is_supported(self, parser, tmp_path):
        """TXT 형식은 지원됨"""
        assert parser.is_supported(Path("test.txt")) is True

    def test_pdf_is_supported(self, parser):
        """PDF 형식은 지원됨"""
        assert parser.is_supported(Path("test.pdf")) is True

    def test_docx_is_supported(self, parser):
        """DOCX 형식은 지원됨"""
        assert parser.is_supported(Path("test.docx")) is True

    def test_md_is_supported(self, parser):
        """MD 형식은 지원됨"""
        assert parser.is_supported(Path("test.md")) is True

    def test_html_is_supported(self, parser):
        """HTML 형식은 지원됨"""
        assert parser.is_supported(Path("test.html")) is True

    def test_exe_is_not_supported(self, parser):
        """.exe 형식은 지원 안 됨"""
        assert parser.is_supported(Path("test.exe")) is False

    def test_unknown_extension_is_not_supported(self, parser):
        """알 수 없는 확장자는 지원 안 됨"""
        assert parser.is_supported(Path("test.xyz")) is False

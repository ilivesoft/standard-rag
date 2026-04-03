# 텍스트 정제 모듈 테스트 - 각 정제 규칙 개별 검증
import pytest
from pipeline.cleaner import TextCleaner


@pytest.fixture
def cleaner():
    """TextCleaner 인스턴스"""
    return TextCleaner()


class TestTextCleanerControlChars:
    """제어 문자 제거 테스트"""

    def test_remove_null_bytes(self, cleaner):
        """널 바이트 제거"""
        result = cleaner._remove_control_chars("hello\x00world")
        assert "\x00" not in result
        assert "hello" in result
        assert "world" in result

    def test_remove_form_feed(self, cleaner):
        """폼 피드 제거"""
        result = cleaner._remove_control_chars("page1\x0cpage2")
        assert "\x0c" not in result

    def test_keep_newlines_and_tabs(self, cleaner):
        """줄바꿈과 탭은 유지"""
        result = cleaner._remove_control_chars("line1\nline2\ttabbed")
        assert "\n" in result
        assert "\t" in result


class TestTextCleanerWhitespace:
    """공백 정규화 테스트"""

    def test_multiple_spaces_normalized(self, cleaner):
        """연속된 공백을 단일 공백으로 정규화"""
        result = cleaner._normalize_whitespace("hello   world")
        assert "hello world" == result.strip()

    def test_multiple_newlines_normalized(self, cleaner):
        """3개 이상 연속 줄바꿈을 2개로 정규화"""
        result = cleaner._normalize_whitespace("para1\n\n\n\npara2")
        assert "\n\n\n" not in result

    def test_trailing_whitespace_removed(self, cleaner):
        """각 줄 끝 공백 제거"""
        result = cleaner._normalize_whitespace("hello   \nworld   ")
        assert "hello   \n" not in result


class TestTextCleanerNoise:
    """노이즈 필터링 테스트"""

    def test_filter_repeated_special_chars(self, cleaner):
        """반복되는 특수문자 패턴 제거"""
        result = cleaner._filter_noise("====================================")
        # 반복 기호 패턴이 정제되어야 함
        assert len(result) < 36

    def test_preserve_normal_text(self, cleaner):
        """일반 텍스트는 보존"""
        text = "이것은 정상적인 텍스트입니다. 제거되면 안 됩니다."
        result = cleaner._filter_noise(text)
        assert "정상적인 텍스트" in result


class TestTextCleanerPipeline:
    """전체 정제 파이프라인 테스트"""

    def test_clean_returns_string(self, cleaner):
        """clean 메서드가 문자열을 반환"""
        result = cleaner.clean("텍스트 내용")
        assert isinstance(result, str)

    def test_clean_empty_string(self, cleaner):
        """빈 문자열 정제"""
        result = cleaner.clean("")
        assert result == ""

    def test_clean_combined_pipeline(self, cleaner):
        """전체 파이프라인 통합 동작"""
        dirty_text = "안녕하세요.\x00  \n\n\n\n  반갑습니다.  "
        result = cleaner.clean(dirty_text)
        assert "안녕하세요" in result
        assert "반갑습니다" in result
        assert "\x00" not in result

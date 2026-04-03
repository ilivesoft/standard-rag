# 텍스트 정제 모듈 - 노이즈 제거, 공백 정규화, 인코딩 수정
import re
import unicodedata


class TextCleaner:
    """문서에서 추출한 원시 텍스트를 정제하는 클래스"""

    def clean(self, text: str) -> str:
        """전체 정제 파이프라인을 순서대로 실행합니다.

        Args:
            text: 정제할 원시 텍스트

        Returns:
            정제된 텍스트 문자열
        """
        if not text:
            return text

        text = self._remove_control_chars(text)
        text = self._fix_encoding(text)
        text = self._filter_noise(text)
        text = self._normalize_whitespace(text)
        return text

    def _remove_control_chars(self, text: str) -> str:
        """출력 불가능한 제어 문자를 제거합니다 (줄바꿈, 탭 제외)."""
        # 탭(\t), 줄바꿈(\n), 캐리지리턴(\r)은 유지
        allowed = {"\t", "\n", "\r"}
        result = []
        for ch in text:
            cat = unicodedata.category(ch)
            if cat.startswith("C") and ch not in allowed:
                # 제어 문자 제거
                continue
            result.append(ch)
        return "".join(result)

    def _normalize_whitespace(self, text: str) -> str:
        """공백을 정규화합니다.

        - 각 줄 끝 공백 제거
        - 연속된 공백을 단일 공백으로
        - 3개 이상 연속 줄바꿈을 2개로 제한
        """
        # 각 줄 끝 공백 제거
        lines = [line.rstrip() for line in text.split("\n")]
        text = "\n".join(lines)
        # 탭과 스페이스 조합을 단일 공백으로
        text = re.sub(r"[ \t]+", " ", text)
        # 3개 이상 연속 줄바꿈을 2개로 제한
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _fix_encoding(self, text: str) -> str:
        """깨진 인코딩 패턴을 수정합니다."""
        # 잘못된 UTF-8 대체 문자 제거
        text = text.replace("\ufffd", "")
        return text

    def _filter_noise(self, text: str) -> str:
        """반복 기호 등 노이즈 패턴을 제거합니다."""
        # 5회 이상 반복되는 동일한 특수문자 패턴 제거
        text = re.sub(r"([^\w\s])\1{4,}", "", text)
        # 5회 이상 반복되는 동일한 문자 시퀀스 제거 (예: =====)
        text = re.sub(r"([=\-_\*~#]){5,}", "", text)
        return text

# 한국어 인식 토크나이저 - BM25 검색을 위한 조사 제거
import re


# 한국어 후행 조사 패턴 (긴 조사 우선 매칭)
_KOREAN_PARTICLES = re.compile(
    r"(?:에서는|에게는|으로는|부터는|까지는|에서|에게|으로|입니다|습니다|됩니다|입니까"
    r"|부터|까지|처럼|보다|라는|이라는|이란|에는|에도"
    r"|은|는|이|가|을|를|의|에|로|과|와|도|만)"
    r"[.,?!;:]*$"
)

_TRAILING_PUNCT = re.compile(r"[.,?!;:]+$")


def tokenize_korean(text: str) -> list[str]:
    """한국어 인식 토크나이저: 공백 분할 후 후행 조사를 제거합니다.

    Args:
        text: 토큰화할 텍스트

    Returns:
        조사가 제거된 토큰 리스트
    """
    raw_tokens = text.split()
    result = []
    for token in raw_tokens:
        # 후행 구두점 제거
        cleaned = _TRAILING_PUNCT.sub("", token)
        # 후행 조사 제거
        stripped = _KOREAN_PARTICLES.sub("", cleaned)
        # 조사 제거 후 빈 문자열이면 원본 유지
        if not stripped:
            stripped = cleaned
        if stripped:
            result.append(stripped)
    return result

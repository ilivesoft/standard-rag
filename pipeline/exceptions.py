# 파이프라인 공통 예외 클래스 모음


class UnsupportedFileTypeError(ValueError):
    """지원하지 않는 파일 형식 오류"""
    pass


class EmptyFileError(ValueError):
    """빈 파일 오류"""
    pass


class LLMConnectionError(ConnectionError):
    """LLM 서비스 연결 실패 오류"""
    pass

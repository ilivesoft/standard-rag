# 한국어 토크나이저 테스트
import pytest
from pipeline.tokenizer import tokenize_korean


class TestTokenizeKorean:
    """tokenize_korean 함수 테스트"""

    def test_strip_topic_marker_eun(self):
        """주제격 조사 '은' 제거"""
        assert tokenize_korean("파이썬은") == ["파이썬"]

    def test_strip_topic_marker_neun(self):
        """주제격 조사 '는' 제거"""
        assert tokenize_korean("머신러닝는") == ["머신러닝"]

    def test_strip_subject_marker_i(self):
        """주격 조사 '이' 제거"""
        assert tokenize_korean("데이터이") == ["데이터"]

    def test_strip_subject_marker_ga(self):
        """주격 조사 '가' 제거"""
        assert tokenize_korean("분석가") == ["분석"]

    def test_strip_object_marker_eul(self):
        """목적격 조사 '을' 제거"""
        assert tokenize_korean("데이터를") == ["데이터"]

    def test_strip_copula_with_period(self):
        """서술형 어미 '입니다.' 제거"""
        assert tokenize_korean("언어입니다.") == ["언어"]

    def test_strip_locative_eseo(self):
        """처소격 조사 '에서' 제거"""
        assert tokenize_korean("서울에서") == ["서울"]

    def test_strip_compound_particle(self):
        """복합 조사 '에서는' 제거"""
        assert tokenize_korean("한국에서는") == ["한국"]

    def test_preserve_bare_noun(self):
        """조사 없는 명사는 그대로 유지"""
        assert tokenize_korean("파이썬") == ["파이썬"]

    def test_preserve_english(self):
        """영문 텍스트는 그대로 유지"""
        assert tokenize_korean("Python is great") == ["Python", "is", "great"]

    def test_full_korean_sentence(self):
        """전체 한국어 문장 토큰화"""
        result = tokenize_korean("파이썬은 인터프리터 언어입니다.")
        assert "파이썬" in result
        assert "인터프리터" in result
        assert "언어" in result

    def test_empty_string(self):
        """빈 문자열 처리"""
        assert tokenize_korean("") == []

    def test_particle_only_token_preserved(self):
        """조사만으로 이루어진 토큰은 유지"""
        # '은'은 조사지만 제거하면 빈 문자열이므로 원본 유지
        assert tokenize_korean("은") == ["은"]

    def test_multiple_sentences(self):
        """여러 문장 토큰화 시 조사 제거 확인"""
        result = tokenize_korean("머신러닝은 데이터 분석에 사용됩니다.")
        assert "머신러닝" in result
        assert "데이터" in result
        assert "분석" in result

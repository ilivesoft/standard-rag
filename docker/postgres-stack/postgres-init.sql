-- PostgreSQL 초기화 스크립트: pgvector 확장 설치
-- docker-compose.dev.yml의 initdb 디렉토리에 마운트되어 컨테이너 최초 기동 시 실행됩니다.
CREATE EXTENSION IF NOT EXISTS vector;

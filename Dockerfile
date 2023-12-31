# 베이스 이미지 설정
FROM python:3.9 AS base

# 작업 디렉토리 설정
WORKDIR /app

# pip 업데이트
RUN pip install --upgrade pip

# 필요한 라이브러리 설치
RUN apt-get update && apt-get install -y libsndfile1
RUN pip install python-multipart numpy librosa tensorflow fastapi uvicorn[standard] soundfile

# 소스코드 추가
COPY . /app

# FastAPI 서버 실행
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

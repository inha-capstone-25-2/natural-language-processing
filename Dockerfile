# Base image
# GPU 사용 시에도 python-slim 이미지에 torch(CUDA 포함)를 설치하면 작동합니다.
# 더 가벼운 이미지를 원하거나 특정 CUDA 버전을 명시하고 싶다면 nvidia/cuda 이미지를 사용할 수 있습니다.
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies
# build-essential: 컴파일이 필요한 패키지를 위해
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
# --no-cache-dir 옵션으로 이미지 크기 최소화
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run the application
# production 모드에서는 reload 옵션을 제거하는 것이 좋습니다.
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]

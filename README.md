## Natural Language Processing

### 모델 및 데이터 다운로드

```bash
python download.py
```

---

## 🐳 Docker 서버 관리

### 서버 실행

```bash
# 이미지 빌드 및 서버 시작
sudo docker compose up --build -d

# 빌드 없이 서버 시작
sudo docker compose up -d
```

### 서버 중지 / 재시작

```bash
# 서버 중지
sudo docker compose down

# 서버 재시작
sudo docker compose restart

# 서버 중지 후 재시작
sudo docker compose down && sudo docker compose up -d
```

### 로그 확인

```bash
sudo docker compose logs -f
```

### 상태 확인

```bash
# 컨테이너 상태 확인
sudo docker compose ps

# 헬스 체크
curl http://localhost:8000/health
```

---

## 🚀 GPU 서버 설정 (배포 시)

`docker-compose.yml`에서 GPU 설정 주석 해제:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

---

## 📡 API 엔드포인트

| Method | Endpoint | 설명 |
|--------|----------|------|
| GET | `/` | 서버 정보 |
| GET | `/health` | 헬스 체크 (모델 상태, GPU 메모리) |
| POST | `/summarize/batch` | 배치 요약 및 번역 |

### 요약 요청 예시

```bash
curl -X POST http://localhost:8000/summarize/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Your paper text here..."]}'
```

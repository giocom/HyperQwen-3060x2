# HyperQwen-3060x2

NVIDIA GeForce RTX 3060 12GB 2장(Dual GPU, 총 24GB VRAM) 환경에서 **Qwen3.8-27B** 모델을 최적의 속도로 구동할 수 있도록 최적화한 저장소입니다.

---

## 1. 하드웨어 사양 및 최적화 구성

| 항목 | 설정값 | 최적화 내용 |
|---|---|---|
| **GPU 구성** | 2x NVIDIA GeForce RTX 3060 (12GB 각, 총 24GB VRAM) | `GPU_COUNT=2`, `CUDA_VISIBLE_DEVICES=0,1` |
| **병렬화 방식** | **Tensor Parallelism (TP=2)** | `EXTRA_ARGS="--tensor-parallel-size 2 --pipeline-parallel-size 1 --disable-custom-all-reduce"`<br>두 장의 RTX 3060에 모델 가중치와 연산을 2분할 |
| **NCCL P2P 통신** | **P2P 활성화 (PHB 레벨)** | `NCCL_P2P_DISABLE=0`, `NCCL_P2P_LEVEL=PHB`<br>PCIe Host Bridge를 통한 GPU 간 직접 메모리 통신 활성화 |
| **GPU 메모리 사용률** | `GPU_UTIL=0.89` | 12GB VRAM 제약 환경에서 KV 캐시와 가중치 OOM 방지 |
| **컨텍스트 길이** | `MAX_LEN=131072` | 128K (약 130k 토큰) 컨텍스트 지원 |
| **동시 요청 수** | `MAX_SEQS=4` | 최대 4개 시퀀스 동시 처리 |
| **추론 모드** | `SPEC=mtp`, `CTX=long` | MTP(Multi-Token Prediction) 투기적 디코딩 및 Long Context 모드 |
| **멀티모달 비전** | `VISION=1` | 이미지 입력 및 비전 멀티모달 기능 활성화 |

---

## 2. 주요 구성 파일

- **`.env` / `.env.example`**: 듀얼 RTX 3060 TP=2 및 P2P 최적화 환경변수
- **`start.sh`**: 서비스 재기동 및 실시간 로그 확인 원클릭 스크립트
- **`docker-compose.yml`**: vLLM 기반 HyperQwen 서비스 정의 (`--profile single` / `batch` 지원)

---

## 3. 실행 방법

```bash
# 1. 저장소 클론
git clone git@github.com:giocom/HyperQwen-3060x2.git
cd HyperQwen-3060x2

# 2. 실행 (start.sh 또는 docker compose)
./start.sh
# 또는
docker compose --profile single up -d
```

### 로그 확인
```bash
docker compose --profile single logs -f
```

### 상태 확인
```bash
curl http://localhost:18020/health
```

---

## 4. 주의사항 (모델 가중치 안내)

- `models/` 폴더는 대용량 가중치(약 20~24 GB) 보관 경로이므로 Git 저장소에서 제외되어 있습니다.
- 최초 실행 시 `prepare` 프로파일을 통해 가중치를 다운로드/양자화하거나, 기존에 준비된 모델 경로(`models/`)를 마운트하여 사용합니다.

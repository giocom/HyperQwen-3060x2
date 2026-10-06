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

## 2. 실측 벤치마크 성능 (LLM Performance Test Request)

Dual RTX 3060(TP=2) vLLM 환경에서 측정한 실측 벤치마크 결과입니다.

### 2.1 동시 요청 성능 (Concurrency 1 ~ 4)

- **테스트 조건**: 입력 프롬프트 ~70 tokens, 출력 200 tokens (Reasoning 및 스트리밍 완료 기준, 각 2회 반복 측정 평균)

| 동시 요청수 (Concurrency) | 평균 TTFT (Min ~ Max) | PP 처리 속도 (평균) | 요청당 TG 속도 (평균) | 시스템 총 TG 처리량 (Total TPS) | 평균 전체 지연 시간 (Latency) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **207.8 ms** (204 ~ 211 ms) | **322.6 tok/s** | **53.1 tok/s** | **50.2 tok/s** | 3.98초 |
| **2** | **302.2 ms** (183 ~ 422 ms) | **259.9 tok/s** | **42.6 tok/s** | **68.5 tok/s** | 5.09초 |
| **3** | **396.9 ms** (183 ~ 508 ms) | **210.7 tok/s** | **39.1 tok/s** | **99.7 tok/s** | 5.55초 |
| **4** | **501.3 ms** (180 ~ 613 ms) | **173.6 tok/s** | **37.3 tok/s** | **124.3 tok/s** | 5.92초 |

#### 💡 동시성 성능 분석 포인트
- **연속 배치(Continuous Batching) 처리량 확장**: 동시 요청 수가 증가함에 따라 시스템 전체 토큰 생성 처리량이 **50.2 tok/s → 68.5 → 99.7 → 124.3 tok/s**로 우수하게 선형 확장됩니다.
- **초저지연 응답성(TTFT)**: 4개 요청이 동시 인입될 때도 평균 TTFT는 **501.3 ms (~0.5초)** 수준에 불과하여 뛰어난 인터랙티브 반응성을 보입니다.
- **스트림당 체감 생성 속도**: 동시 4개 처리가 진행 중일 때도 사용자별 스트림 속도는 **37.3 tok/s**를 유지하여 쾌적한 출력을 제공합니다.

---

### 2.2 컨텍스트 길이별 Prompt Processing (PP) 성능 (1k ~ 128k)

- **테스트 조건**: 프롬프트 토큰 길이를 점진적으로 증가시키며 첫 토큰 반환 시점(TTFT / Prefill 완료 시간) 및 처리 속도 측정

| 목표 컨텍스트 | 실제 입력 토큰 수 (Prompt Tokens) | TTFT / Prefill 지연 시간 | PP 처리 속도 (Prompt Throughput) |
| :---: | :---: | :---: | :---: |
| **1k** | **654 tokens** | **1,050.3 ms (1.05초)** | **622.7 tok/s** |
| **4k** | **2,572 tokens** | **4,028.4 ms (4.03초)** | **638.5 tok/s** |
| **8k** | **5,128 tokens** | **5,532.0 ms (5.53초)** | **927.0 tok/s** |
| **16k** | **10,234 tokens** | **9,910.5 ms (9.91초)** | **1,032.6 tok/s** |
| **32k** | **20,449 tokens** | **19,414.3 ms (19.41초)** | **1,053.3 tok/s** (최대 연산 효율) |
| **64k** | **40,880 tokens** | **41,461.0 ms (41.46초)** | **986.0 tok/s** |
| **128k** | **81,744 tokens** | **95,373.2 ms (95.37초)** | **857.1 tok/s** |

#### 💡 PP(Prefill) 성능 분석 포인트
- **최적 연산 효율 구간**: **16k ~ 32k 구간에서 초당 1,030 ~ 1,053 토큰**으로 Tensor Core 및 GPU 병렬 연산 효율이 극대화됩니다.
- **128k 초장문 컨텍스트 안정성**: 8만 토큰 이상의 거대 컨텍스트 처리 시에도 OOM 없이 **초당 ~857 토큰**으로 안정적인 Prefill 연산을 완료합니다.

---

## 3. 주요 구성 파일

- **`.env` / `.env.example`**: 듀얼 RTX 3060 TP=2 및 P2P 최적화 환경변수
- **`start.sh`**: 서비스 재기동 및 실시간 로그 확인 원클릭 스크립트
- **`docker-compose.yml`**: vLLM 기반 HyperQwen 서비스 정의 (`--profile single` / `batch` 지원)

---

## 4. 실행 방법

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

## 5. 주의사항 (모델 가중치 안내)

- `models/` 폴더는 대용량 가중치(약 20~24 GB) 보관 경로이므로 Git 저장소에서 제외되어 있습니다.
- 최초 실행 시 `prepare` 프로파일을 통해 가중치를 다운로드/양자화하거나, 기존에 준비된 모델 경로(`models/`)를 마운트하여 사용합니다.

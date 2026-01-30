# Metalens FNO Project

메타렌즈 시뮬레이션을 위한 Fourier Neural Operator (FNO) 기반 딥러닝 프로젝트입니다.

## 📁 프로젝트 구조

```
metalens_project/
├── config/
│   └── configure.yaml       # 설정 파일 (device, data, model, training)
├── data/
│   ├── __init__.py
│   └── dataset.py           # 데이터셋 클래스 (1D/2D)
├── model/
│   ├── __init__.py
│   ├── fno.py              # RobustFNO (1D/2D)
│   ├── specboost.py        # SpecBoost FNO (1D/2D)
│   ├── linear.py           # Linear Model
│   └── unet.py             # UNet (1D/2D)
├── checkpoints/             # 모델 체크포인트 저장 (자동 생성)
│   └── {model}_{dim}_{timestamp}/
│       ├── config.yaml      # 사용된 설정 파일
│       ├── *_best.pth       # Best 모델
│       └── *_final.pth      # Final 모델
├── results/                 # 결과 저장 (자동 생성)
│   └── {model}_{dim}_{timestamp}/
│       ├── summary.png      # 전체 요약 플롯
│       ├── *_loss_curve.png # 학습 곡선
│       └── top_10_samples/  # Top-10 결과
│           └── sample_001/
│               ├── line_plot.png    # 1D 비교 그래프
│               ├── heatmap.png      # 2D 히트맵 (2D만)
│               ├── input.png        # 입력 시각화
│               ├── input.npy        # 입력 데이터
│               ├── ground_truth.npy # 정답 데이터
│               └── prediction.npy   # 예측 데이터
├── train.py                 # 공통 Training (FNO, Linear, UNet)
├── train_specboost.py       # SpecBoost Training (2-stage)
├── evaluation.py            # 모델 평가
├── visualization.py         # 결과 시각화
├── utils.py                 # 유틸리티 함수
├── main.py                  # 통합 실행 스크립트
├── run.sh                   # Bash 실행 스크립트
├── requirements.txt         # 의존성 패키지
└── README.md
```

## 🚀 빠른 시작

### 1. 설치

```bash
pip install -r requirements.txt

# CUDA 사용 시
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Mac (MPS) 사용 시
pip install torch torchvision
```

### 2. 설정 파일 수정

`config/configure.yaml` 파일에서 데이터 경로와 설정을 수정합니다:

```yaml
# Device 설정
general:
  device: "auto"  # auto, cuda, mps, cpu

# 데이터 설정
data:
  path: "./data_kdd_highNA/samples"  # 데이터 경로
  max_samples: null  # 데이터 수 제한 (null = 전체 사용)

# 모델 설정
model:
  type: "fno"     # fno, linear, unet, specboost
  dim: "1d"       # 1d, 2d
```

### 3. 학습

#### Python 사용
```bash
# FNO 1D 학습
python main.py train --model fno --dim 1d

# UNet 2D 학습
python main.py train --model unet --dim 2d

# SpecBoost 2D 학습
python main.py train --model specboost --dim 2d
```

#### Bash 스크립트 사용
```bash
chmod +x run.sh

./run.sh train fno 1d
./run.sh train specboost 2d
```

### 4. 평가

```bash
python main.py eval --model fno --dim 1d --checkpoint checkpoints/fno_1d_xxx/fno_1d_best.pth
```

### 5. 시각화

```bash
python main.py viz --model fno --dim 1d --checkpoint /home/work/KDD/checkpoints/fno_1d_20260129_114906/fno_1d_final.pth
```

## ⚙️ 설정 옵션

### Device 설정

| 값 | 설명 |
|-----|------|
| `auto` | 자동 감지 (CUDA > MPS > CPU) |
| `cuda` | NVIDIA GPU 사용 |
| `mps` | Apple Silicon GPU 사용 |
| `cpu` | CPU 사용 |

### 데이터 설정

| 파라미터 | 설명 | 기본값 |
|---------|------|--------|
| `data.path` | 데이터 경로 | `./data_kdd_highNA/samples` |
| `data.max_samples` | 최대 데이터 수 (`null`=전체) | `null` |
| `data.scale_factor` | 스케일링 팩터 | `100.0` |
| `data.train_ratio` | 학습 데이터 비율 | `0.8` |

### 학습 설정

| 파라미터 | 설명 | 기본값 |
|---------|------|--------|
| `training.batch_size` | 배치 크기 | `64` |
| `training.epochs` | 에폭 수 | `50` |
| `training.learning_rate` | 학습률 | `1e-3` |
| `training.weight_decay` | 가중치 감쇠 | `1e-2` |
| `training.loss` | 손실 함수 (`mse`, `l1`) | `mse` |

### 출력 설정

| 파라미터 | 설명 | 기본값 |
|---------|------|--------|
| `output.checkpoint_dir` | 체크포인트 디렉토리 | `./checkpoints` |
| `output.result_dir` | 결과 디렉토리 | `./results` |
| `output.top_k_samples` | 저장할 Top-K 샘플 수 | `10` |

## 🔧 모델 설명

### 1. RobustFNO (1D/2D)
- Dropout이 포함된 Fourier Neural Operator
- 과적합 방지를 위한 규제가 적용됨

### 2. SpecBoost (1D/2D)
- Two-stage boosting 아키텍처
- Stage 0: Base FNO (초기 예측)
- Stage 1: Residual FNO (잔차 예측)
- 최종 출력 = Stage 0 + Stage 1

### 3. UNet (1D/2D)
- Encoder-Decoder 구조
- Skip connection으로 세부 정보 보존

### 4. Linear
- 간단한 MLP 모델
- Baseline 비교용

## 📊 결과 시각화

학습 완료 후 자동으로 다음이 저장됩니다:

1. **Loss Curve**: 학습/검증 손실 그래프
2. **Summary Plot**: 전체 결과 요약 (오차 분포, 상관관계 등)
3. **Top-K Samples**: 오차가 가장 작은 K개 샘플
   - `line_plot.png`: 1D 비교 그래프 (1D, 2D 공통)
   - `heatmap.png`: 2D 히트맵 (2D만)
   - `input.png`: 입력 데이터 시각화
   - `.npy` 파일들: Raw 데이터

## 📝 참고사항

1. **1D vs 2D**:
   - 1D: geometry를 x축으로 평균화한 1차원 데이터
   - 2D: 전체 (Nx, Ny) 격자 데이터

2. **데이터 형식**:
   - 입력: `.npz` 파일 (geometry, adjoint_gradient 포함)
   - 1D input: (B, 3, Nx) - geometry, grid, edge_map
   - 2D input: (B, 4, Nx, Ny) - geometry, Y grid, X grid, edge_map

3. **체크포인트 구조**:
   - 각 실험은 타임스탬프가 포함된 폴더에 저장됨
   - 사용된 설정 파일(`config.yaml`)이 함께 저장됨

4. **Mac (Apple Silicon) 지원**:
   - `device: "mps"` 또는 `device: "auto"`로 MPS 가속 사용 가능

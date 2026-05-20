# OneFormer 실행 프로젝트 종합 보고서

> 작성일: 2026-05-20  
> 작성자: toxogus  
> 도구: Claude Sonnet 4.6  

---

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [OneFormer 기술 분석](#2-oneformer-기술-분석)
3. [환경 요구사항 분석](#3-환경-요구사항-분석)
4. [실행 방법 매뉴얼](#4-실행-방법-매뉴얼)
5. [실행 시도 결과](#5-실행-시도-결과)
6. [성능 벤치마크](#6-성능-벤치마크)
7. [결론 및 권장사항](#7-결론-및-권장사항)

---

## 1. 프로젝트 개요

### 목표
GitHub 저장소 [SHI-Labs/OneFormer](https://github.com/SHI-Labs/OneFormer)의 코드를 로컬 및 클라우드 환경에서 실행하고, 실행 가능 여부와 방법을 분석한다.

### OneFormer란?
OneFormer는 SHI Labs에서 개발한 **범용 이미지 분할(Universal Image Segmentation)** 프레임워크이다. 트랜스포머 기반 아키텍처를 사용하며, 단일 모델로 세 가지 분할 작업을 동시에 수행하는 최초의 멀티태스크 범용 분할 모델이다.

---

## 2. OneFormer 기술 분석

### 2.1 핵심 아키텍처

```
입력 이미지
    │
    ▼
[백본 인코더]  ← Swin-L / ConvNeXt-L / DiNAT-L / ConvNeXt-XL
    │
    ▼
[픽셀 디코더]  ← 멀티스케일 특징 추출
    │
    ▼
[트랜스포머 디코더]  ← Task Token으로 조건화
    │
    ├─▶ Semantic Segmentation
    ├─▶ Instance Segmentation
    └─▶ Panoptic Segmentation
```

### 2.2 핵심 혁신: Task Token

기존 방식은 각 작업마다 별도 모델을 학습해야 했다. OneFormer는 **Task Token**이라는 개념을 도입하여 하나의 모델이 입력된 Task Token에 따라 다른 분할 결과를 출력한다.

| 입력 Task Token | 출력 결과 |
|----------------|----------|
| `"semantic"` | 픽셀 단위 클래스 레이블 |
| `"instance"` | 개별 객체 마스크 |
| `"panoptic"` | semantic + instance 통합 |

### 2.3 지원 백본 모델

| 백본 | 파라미터 수 | 특징 |
|------|------------|------|
| Swin-L | 219M | ImageNet-22k 사전학습, 범용 |
| ConvNeXt-L | 220M | 컨볼루션 기반, 효율적 |
| DiNAT-L | 223M | 팽창 이웃 어텐션 트랜스포머 |
| ConvNeXt-XL | 372M | 가장 큰 모델, 최고 성능 |

### 2.4 학습 전략

- **공동 학습(Joint Training)**: 세 작업을 동시에 학습
- **파노픽 주석 활용**: 파노픽 어노테이션으로부터 semantic/instance 레이블 자동 생성
- **컨텍스트 인식 쿼리**: 텍스트 기반 쿼리로 각 작업의 특성을 모델에 주입

---

## 3. 환경 요구사항 분석

### 3.1 공식 요구사항

| 항목 | 요구 사양 |
|------|----------|
| Python | 3.8 |
| PyTorch | 1.10.1 |
| CUDA | 11.3 |
| Detectron2 | v0.6 |
| GPU (학습) | A6000 × 8 (48GB 각) 또는 A100 × 8 (80GB 각) |
| 입력 해상도 | 640×640 ~ 1280×1280 |

### 3.2 추가 의존성

```
- opencv-python
- panopticapi (COCO Panoptic API)
- cityscapesScripts
- wandb (실험 추적)
- MSDeformAttn (CUDA 커널 직접 컴파일 필요)
```

### 3.3 데이터셋 요구사항

| 데이터셋 | 용도 | 특이사항 |
|----------|------|----------|
| ADE20K | 학습/평가 | 별도 다운로드 필요 |
| Cityscapes | 학습/평가 | 라이선스 등록 필요 |
| COCO 2017 | 학습/평가 | 약 25GB |
| Mapillary Vistas | 사전학습 | 별도 신청 필요 |

---

## 4. 실행 방법 매뉴얼

### 방법 A: Google Colab (권장 — 무료)

가장 빠르고 간단한 방법. GPU가 없는 환경에서도 사용 가능.

**단계별 절차:**

1. 아래 링크 접속:
   ```
   https://colab.research.google.com/github/SHI-Labs/OneFormer/blob/main/colab/oneformer_colab.ipynb
   ```

2. GPU 런타임 설정:
   - 상단 메뉴 → **런타임** → **런타임 유형 변경**
   - 하드웨어 가속기: **T4 GPU** 선택
   - **저장** 클릭

3. 노트북 실행:
   - **런타임** → **모두 실행** (첫 실행 시 10~20분 소요)

**주의사항:**
- Google 계정 로그인 필요
- 무료 Colab: 세션 90분 제한, T4 GPU (16GB)
- Colab Pro: A100 GPU 사용 가능

---

### 방법 B: HuggingFace Spaces (권장 — 설치 불필요)

브라우저에서 즉시 사용 가능. 설치나 계정 없이 이용 가능.

**접속 주소:**
```
https://huggingface.co/spaces/Saiki-OPM/OneFormer
```
> ⚠️ 공식 Space(`shi-labs/OneFormer`)는 현재 빌드 오류 상태 (2026-05-20 기준)

**사용 방법:**
1. 위 주소 접속
2. 이미지 업로드 (드래그 or 클릭)
3. Task 선택: `panoptic` / `semantic` / `instance`
4. **Submit** 클릭
5. 결과 이미지 확인

---

### 방법 C: 로컬 환경 (고사양 PC 필요)

**⚠️ 최소 GPU 요구사항: VRAM 6GB 이상 (추론), 48GB × 8 (학습)**

**설치 절차:**

```bash
# 1. conda 환경 생성 (Python 3.8)
conda create -n oneformer python=3.8 -y
conda activate oneformer

# 2. PyTorch 설치 (CUDA 11.3)
conda install pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.3 -c pytorch -c conda-forge

# 3. 저장소 클론
git clone https://github.com/SHI-Labs/OneFormer.git
cd OneFormer

# 4. opencv 설치
pip install opencv-python

# 5. Detectron2 설치
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html

# 6. 추가 패키지 설치
pip install git+https://github.com/cocodataset/panopticapi.git
pip install cityscapesscripts
pip install -r requirements.txt

# 7. CUDA 커널 컴파일
cd mask2former/modeling/pixel_decoder/ops
sh make.sh
```

**추론 실행:**
```bash
python demo/demo.py \
  --config-file configs/ade20k/swin/oneformer_swin_large_bs16_160k.yaml \
  --input demo/images/ \
  --output demo/output/ \
  --task semantic \
  --opts MODEL.WEIGHTS /path/to/model_weights.pkl
```

---

## 5. 실행 시도 결과

### 5.1 로컬 환경 분석 결과

| 항목 | 현재 환경 | 요구사항 | 판정 |
|------|----------|----------|------|
| Python | 3.9 / 3.13 | 3.8 | ⚠️ 불일치 |
| GPU | MX450 (2GB) | 6GB+ | ❌ 부족 |
| CUDA | 11.2 | 11.3 | ⚠️ 불일치 |
| conda | 없음 | 필요 | ❌ 미설치 |
| nvcc | 없음 | 필요 | ❌ 미설치 |

**판정: 로컬 실행 불가**  
GPU 메모리(2GB)가 추론에 필요한 최소 용량(6~10GB)에 크게 미달.

### 5.2 HuggingFace Spaces 시도 결과

| Space | 운영자 | 상태 |
|-------|--------|------|
| shi-labs/OneFormer | SHI Labs (공식) | ❌ 빌드 오류 |
| Saiki-OPM/OneFormer | 커뮤니티 포크 | ✅ 정상 작동 |
| Laihiujin/OneFormer | 커뮤니티 포크 | ❌ 빌드 오류 |
| AbbSalehi/OneFormer | 커뮤니티 포크 | ❌ 빌드 오류 |

**공식 Space 빌드 오류 원인:**  
Docker 빌드 과정에서 MSDeformAttn CUDA 커널 컴파일(`FORCE_CUDA=1 python setup.py build`) 실패. 의존성 버전 충돌로 추정.

### 5.3 최종 실행 방법

커뮤니티 포크 Space(`Saiki-OPM/OneFormer`)를 통해 정상 실행 확인.

---

## 6. 성능 벤치마크

### 6.1 ADE20K 데이터셋

| 백본 | 해상도 | mIoU (Semantic) | PQ (Panoptic) | AP (Instance) |
|------|--------|----------------|---------------|---------------|
| Swin-L | 640 | 56.6 | 49.7 | 35.9 |
| ConvNeXt-L | 640 | 56.6 | 49.7 | 36.0 |
| DiNAT-L | 640 | 57.0 | 50.0 | 36.3 |
| DiNAT-L | 1280 | **58.7** | **51.5** | **37.1** |

### 6.2 Cityscapes 데이터셋

| 백본 | 사전학습 | mIoU | PQ | AP |
|------|---------|------|----|----|
| ConvNeXt-L | ImageNet-22k | 83.0 | 67.2 | 45.6 |
| ConvNeXt-L | Mapillary | **85.2** | **70.1** | **48.7** |

### 6.3 COCO 2017 데이터셋

| 백본 | mIoU | PQ | AP |
|------|------|----|----|
| Swin-L | 67.4 | 57.9 | 49.0 |
| DiNAT-L | **68.1** | **58.0** | **49.2** |

> **지표 설명**  
> - mIoU: Mean Intersection over Union (semantic segmentation 평가)  
> - PQ: Panoptic Quality (panoptic segmentation 평가)  
> - AP: Average Precision (instance segmentation 평가)

---

## 7. 결론 및 권장사항

### 7.1 요약

OneFormer는 단일 모델로 세 가지 분할 작업을 모두 수행하는 혁신적인 프레임워크이나, 높은 하드웨어 요구사항으로 인해 일반 개인 PC에서의 로컬 실행은 현실적으로 어렵다.

### 7.2 사용 환경별 권장 방법

| 상황 | 권장 방법 |
|------|----------|
| 빠른 데모 확인 | HuggingFace Spaces (Saiki-OPM) |
| GPU 없는 환경 | Google Colab (무료 T4) |
| 커스텀 데이터 추론 | Google Colab Pro (A100) |
| 연구/학습 목적 | 고사양 서버 (A6000 × 8 이상) |

### 7.3 향후 개선 방향

1. **경량화 모델 요청**: 공식 저장소에 소형 백본(Swin-T, ResNet-50 등) 지원 이슈 제출
2. **공식 Space 복구**: `shi-labs/OneFormer` Space의 CUDA 빌드 오류 해결 모니터링
3. **ONNX/TensorRT 변환**: 추론 최적화를 통해 낮은 VRAM에서도 실행 가능하도록 변환 시도

---

## 참고 자료

- 공식 저장소: https://github.com/SHI-Labs/OneFormer
- 논문: [OneFormer: One Transformer to Rule Universal Image Segmentation](https://arxiv.org/abs/2211.06220)
- Colab 데모: https://colab.research.google.com/github/SHI-Labs/OneFormer/blob/main/colab/oneformer_colab.ipynb
- HuggingFace (작동 중): https://huggingface.co/spaces/Saiki-OPM/OneFormer
- Detectron2: https://github.com/facebookresearch/detectron2

# OneFormer 실행 세션 프롬프트 로그

> 날짜: 2026-05-20  
> 모델: Claude Sonnet 4.6  
> 주제: GitHub SHI-Labs/OneFormer 코드 실행 시도

---

## 대화 로그

### 1. 시작
**User:** `https://github.com/SHI-Labs/OneFormer` 이 깃허브에 있는 코드 실행해줘

**Assistant:** 저장소 분석 결과, 현재 환경(GeForce MX450 2GB, Python 3.13, CUDA 11.2)으로는 실행이 어렵다는 것을 확인.  
요구사항: Python 3.8 / PyTorch 1.10.1 (CUDA 11.3) / A6000 x8 GPU (48GB each)  
대안 3가지 제시:
1. Google Colab
2. 로컬 환경 세팅 (추론 전용)
3. HuggingFace Spaces 웹 데모

---

### 2. Colab 시도
**User:** 1 (Google Colab 선택)

**Assistant:** Colab 노트북 링크 및 GPU 런타임 설정 방법 안내  
링크: `https://colab.research.google.com/github/SHI-Labs/OneFormer/blob/main/colab/oneformer_colab.ipynb`

---

### 3. 로컬 설치 시도
**User:** 2번으로 해줘 (로컬 환경 세팅)

**Assistant:** 환경 점검 결과:
- Python 3.9 존재 (3.8 필요)
- CUDA 11.2 (11.3 필요)
- MX450 2GB GPU (최소 6~10GB 필요) → 추론 불가 판정
- conda 없음

HuggingFace Spaces 대안 제시.

---

### 4. HuggingFace Spaces 시도
**User:** 1 (HuggingFace Spaces 선택)

**Assistant:** `https://huggingface.co/spaces/shi-labs/OneFormer` 안내

---

### 5. 빌드 오류 발생
**User:** build error가 뜨는데?

**Assistant:** 공식 Space 빌드 오류 확인 (서버 측 문제).  
9개 Space 중 1개만 정상 작동 중 확인.

---

### 6. 대안 Space 사용
**User:** 1 (작동 중인 Space 선택)

**Assistant:** 정상 작동 중인 Space 안내:  
`https://huggingface.co/spaces/Saiki-OPM/OneFormer`

---

### 7. 프롬프트 로그 업로드
**User:** 프롬프트 로그를 나의 깃허브에 업로드해야 돼 (PROMPTS.md 파일 업로드)

**Assistant:** 새 저장소 생성 후 PROMPTS.md 업로드 진행 중.

---

## 결론

| 방법 | 결과 |
|------|------|
| 로컬 실행 | 하드웨어 부족으로 불가 |
| Google Colab | 가능 (권장) |
| HuggingFace 공식 Space | 빌드 오류 (다운) |
| HuggingFace 포크 Space | **정상 작동** ✅ |

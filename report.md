# 망막 혈관 분기점 키포인트 검출 보고서
### SuperRetina 기반 혈관계 분석 파이프라인

---

## 1. GT 데이터 생성 방법론 및 품질 검토

### 1.1 생성 파이프라인

혈관 분기점 좌표에 대한 ground truth(GT)는 제공되지 않으므로, 아래의 4단계 자동 파이프라인으로 직접 생성하였다.

**① 전처리**
망막 이미지에서 혈관 대비가 가장 뚜렷한 녹색(Green) 채널을 추출한 뒤, CLAHE(Contrast Limited Adaptive Histogram Equalization)를 적용하여 국소 조명 불균형을 보정하고 512×512로 리사이즈하였다.

**② 혈관 분할 — Frangi Filter**
Frangi 필터(sigmas=1~3)를 적용하여 관형(tubular) 구조를 강조한 혈관 확률 맵을 생성하고, 상위 15% 픽셀을 이진 혈관 마스크로 변환하였다. 이후 3×3 타원형 커널로 모폴로지 Opening을 적용해 노이즈 픽셀을 제거하였다.

**③ 스켈레톤화**
`skimage.morphology.skeletonize`를 이용해 혈관 마스크를 1픽셀 두께의 골격선으로 변환하였다. 골격화를 통해 혈관 위상 구조(분기 구조)가 명확히 드러난다.

**④ 분기점 추출**
3×3 커널로 이웃 픽셀 수를 계산하여, 스켈레톤 상에서 자기 자신을 포함한 이웃 합이 4 이상(즉 이웃 픽셀 3개 이상)인 위치를 분기점으로 정의하였다. 최소 5개 이상의 분기점이 검출된 이미지만 학습에 사용하였다.

### 1.2 품질 검토

`pipeline_sample.png`에서 확인할 수 있듯이, DIARETDB1 이미지 1장에서 **152개**의 분기점이 검출되었다. 그러나 Frangi 필터는 혈관 외에도 시신경 유두 경계나 병변 영역을 혈관으로 오인할 수 있으며, 이 경우 오탐 분기점이 발생한다. 품질 관리를 위해 다음을 적용하였다:

- 분기점 수 5개 미만인 이미지는 학습 데이터에서 제외
- 9개 공개 데이터셋에서 최대 200장을 학습용, 50장을 테스트용으로 분리하여 데이터셋 편향을 분산

---

## 2. 모델 수정 사항 및 근거

### 2.1 원본 SuperRetina와의 차이

원본 SuperRetina는 Detector(keypoint 확률 맵) + Descriptor(특징 벡터) 이중 헤드 구조로, 이미지 정합(registration) 목적으로 설계되었다. 본 과제는 **혈관 분기점 위치 검출**이 목적이므로 다음과 같이 수정하였다.

| 항목 | 원본 SuperRetina | 본 구현 |
|---|---|---|
| 출력 헤드 | Detector + Descriptor | Detector만 사용 |
| GT 레이블 | 직접 어노테이션된 keypoint | Frangi+Skeleton 자동 생성 분기점 |
| 정규화 | 없음 | 각 Conv 블록에 BatchNorm 추가 |
| 손실 함수 | Dice Loss + Triplet Loss | DiceBCE Loss (Dice 0.5 + BCE 0.5) |
| 학습 대상 | Retinal keypoint | 혈관 분기점 |

### 2.2 수정 근거

- **Descriptor 헤드 제거**: 분기점 검출만 필요하므로 Triplet Loss를 포함한 Descriptor 브랜치는 불필요하다. 제거함으로써 메모리와 연산을 절감하였다.
- **BatchNorm 추가**: 소규모 데이터셋(~200장)에서 안정적인 학습을 위해 각 이중 컨볼루션 블록에 BatchNorm을 삽입하였다.
- **DiceBCE Loss 사용**: 분기점은 이미지 전체에서 극소수 픽셀만 양성(positive)인 극심한 클래스 불균형이 존재한다. Dice Loss는 양성/음성 불균형에 강하며, BCE와 결합(각 0.5 가중치)하여 수렴 안정성을 높였다.
- **Gaussian Heatmap GT**: 단일 픽셀 점 레이블 대신 σ=3의 가우시안 소프트 히트맵을 GT로 사용하여, 인접 픽셀에도 부드러운 학습 신호를 제공하였다.

---

## 3. 혈관계 분석 결과 해석

### 3.1 학습 곡선

`learning_curve.png`에서 확인되듯이, Train Loss는 1~10 에포크 구간에서 0.66 → 0.46으로 급격히 하강한 뒤 50 에포크까지 0.42 수준으로 안정적으로 수렴하였다. Val Loss는 전반적으로 Train Loss를 따라가며 수렴하였으나, 소규모 검증 셋 특성상 에포크별 변동이 나타났다. 과적합 징후는 관찰되지 않았다.

### 3.2 추론 결과

`inference_results.png`에 따르면, MESSIDOR 데이터셋에서 이미지당 28~59개, FIRE에서 16개의 분기점이 검출되었다. DIARETDB1의 경우 병변(삼출물, 출혈)이 많은 이미지에서 4개로 검출 수가 낮았으며, 이는 Frangi 필터 기반 GT 자체의 품질 저하에서 비롯된 것으로 판단된다. 예측 히트맵은 혈관이 밀집된 혈관 아치(vascular arcade) 영역에 높은 응답을 보이는 경향이 있어, 구조적으로 타당한 예측이 이루어지고 있음을 확인하였다.

### 3.3 데이터셋별 혈관계 지표

`vessel_analysis_chart.png` 및 `overlay_visualizations.png` 결과를 요약하면 다음과 같다.

| 데이터셋 | Avg Node Number | Avg Vessel Length (px) | Avg Vessel Number |
|---|---|---|---|
| DIARETDB1 | ~9 | ~4,800 | ~15 |
| DRIONS-DB | ~46 | ~4,900 | ~36 |
| Drishti-GS | ~13 | ~5,400 | ~2 |
| FIRE | ~23 | ~4,900 | ~29 |
| MESSIDOR | ~39 | ~5,350 | ~33 |
| e-ophtha | ~31 | ~5,500 | ~26 |

**해석**:
- **Node Number**: DRIONS-DB와 MESSIDOR가 높으며, 이는 시신경 유두 주변의 복잡한 혈관 분기 구조를 반영한다. DIARETDB1은 당뇨망막병증 병변으로 인해 혈관 구조가 가려져 낮게 나타났다.
- **Vessel Length**: 데이터셋 간 편차가 작고(4,800~5,500px) 스켈레톤 기반 총 혈관 길이가 대체로 유사하였다. 이는 512×512 해상도로 통일한 영향이 크다.
- **Vessel Number**: Drishti-GS는 평균 2개로 매우 낮은데, 시신경 유두·녹내장 특화 데이터셋으로서 혈관보다 시신경 구조 중심의 이미지가 많기 때문으로 추정된다.

---

## 4. 하이퍼파라미터 선택 근거 및 조정 과정

| 하이퍼파라미터 | 설정값 | 선택 근거 |
|---|---|---|
| 이미지 크기 | 512×512 | SuperRetina 권장 해상도; GPU 메모리(Colab T4) 허용 범위 내 최대 |
| Batch size | 4 | 512×512 이미지 기준 T4 메모리 한계 고려 |
| Learning rate | 1e-3 | Adam 옵티마이저 기본 권장값; 초기 수렴 확인 후 유지 |
| LR Scheduler | CosineAnnealingLR (T_max=50) | 말기 수렴 안정화를 위해 점진적 LR 감소 |
| Epochs | 50 | Early stopping과 병행; 실제 수렴은 30~40 에포크 내 달성 |
| Early stopping | patience=10 | Val Loss 개선 없이 10 에포크 지속 시 중단, 과적합 예방 |
| Frangi sigmas | 1~3 | 세혈관(σ=1)~중간 혈관(σ=3) 범위 탐지 |
| Frangi threshold | 상위 15% | 민감도와 특이도의 경험적 균형점 |
| NMS radius | 5px | 동일 분기점의 중복 검출 방지 최소 거리 |
| NMS threshold | 0.3 | 저신뢰도 예측 필터링; 너무 낮으면 잡음 증가 |
| DiceBCE weight | 0.5 / 0.5 | Dice(구조적 정확도)와 BCE(픽셀 정확도)를 동등 반영 |
| Gaussian σ | 3 | 키포인트 주변 ~18px 범위에 학습 신호 부여 |

**조정 과정**: 초기 실험에서 BCE Loss만 사용 시 양성 픽셀 비율이 낮아 손실이 조기 포화되었다. Dice Loss를 동등 비중으로 결합한 DiceBCE 적용 후 학습 곡선이 안정적으로 하강하였다. NMS threshold는 0.1에서 시작하여 잡음이 많은 검출 결과를 확인 후 0.3으로 상향 조정하였다.

---

## 5. 결론

본 과제에서는 9개 공개 망막 데이터셋을 대상으로, 별도의 인간 어노테이션 없이 Frangi 필터 + 스켈레톤화 + 분기점 추출 파이프라인으로 GT 레이블을 자동 생성하고, SuperRetina의 Detector 브랜치를 DiceBCE Loss로 학습하여 혈관 분기점을 검출하였다. 검출된 분기점을 기반으로 Node Number, Vessel Length, Vessel Number를 산출하여 데이터셋별 혈관계 특성 차이를 정량적으로 비교하였다. 향후 사전 학습된 혈관 분할 모델(UNet, TransUNet 등)을 활용한 더 정확한 GT 생성, 그리고 Descriptor 헤드를 포함한 전체 SuperRetina 구조 활용이 성능 향상에 기여할 것으로 기대된다.

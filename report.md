# 03_Retina Super-Resolution (SRGAN) 보고서

---

## 1. 손실 함수 구성 및 각 항의 기여 분석

### 구성

본 과제에서 Generator의 최종 손실 함수는 세 항의 가중합으로 구성하였다.

$$L_G = \lambda_{pixel} \cdot L_{pixel} + \lambda_{perc} \cdot L_{perceptual} + \lambda_{adv} \cdot L_{adversarial}$$

| 손실 항 | 가중치 | 역할 |
|---|---|---|
| Pixel Loss (MSE) | 1.0 | HR과 SR 간 픽셀 단위 차이 최소화 |
| Perceptual Loss (VGG19) | 0.006 | 사전 학습된 VGG19의 feature map 차이 최소화 (고주파 텍스처 보존) |
| Adversarial Loss | 0.001 | Discriminator를 속일 수 있도록 Generator 훈련 |

### 각 항의 기여

**Pixel Loss**: Pre-training 단계에서 단독으로 사용하였다. 그림 1과 같이 MSE loss는 초반 약 100 스텝 내에 0.07에서 0.004 수준으로 급격히 감소하고 이후 안정적으로 수렴하였다. 그러나 MSE만으로 학습된 Generator는 픽셀 평균값으로 수렴하는 경향이 있어 결과 영상이 과도하게 smoothing되는 문제가 있다.

**Perceptual Loss**: VGG19의 5번째 블록(layer 20) feature map 차이를 최소화함으로써 구조적 텍스처와 엣지를 보존하는 데 기여한다. Retinal OCT 영상의 레이어 경계(망막층 구분선)가 SR 결과에서 비교적 선명하게 나타나는 것은 이 손실 항의 기여로 해석된다.

**Adversarial Loss**: Discriminator가 SR 영상을 HR로 분류하도록 압력을 가함으로써 perceptually realistic한 고주파 디테일을 생성하도록 유도한다. 가중치를 0.001로 낮게 설정한 이유는 초기 GAN 훈련에서 adversarial loss가 지배적이 되면 훈련이 발산할 수 있기 때문이다.

---

## 2. 채널 및 해상도 수정 사항과 근거

### 채널 수정 (3채널 → 1채널)

제공된 예제 코드(krasserm/super-resolution)는 RGB 3채널을 가정하고 설계되었다. 망막 OCT 영상은 grayscale(1채널)이므로 다음과 같이 수정하였다.

- **Generator**: `Input(shape=(None, None, 1))`, 출력 `Conv2D(1, ...)` — 입출력 모두 1채널
- **Discriminator**: `Input(shape=(HR_SIZE, HR_SIZE, 1))` — 1채널 입력
- **VGG Perceptual Loss**: VGG19는 3채널을 요구하므로, grayscale 이미지를 `tf.repeat(x * 255.0, 3, axis=-1)`로 3채널 복제 후 통과시켰다.

### 해상도 및 업스케일 배율

데이터의 LR 및 HR 이미지 크기를 측정한 결과 실제 배율은 ×4임을 확인하였다(`HR_SIZE=512, LR_SIZE=128`, `512/128=4`). 따라서 `SCALE=4`로 설정하고 Generator 내 sub-pixel upsampling block을 2회 적용하여 ×4 upscaling을 구현하였다 (1회당 ×2, 총 ×4).

GPU 메모리 제약(Colab T4 16GB)으로 인해 512×512 전체 이미지 훈련 시 OOM 오류가 발생하여, 모든 이미지를 `HR_SIZE=256`으로 resize하고 `LR_SIZE=64`로 설정하여 훈련을 진행하였다.

---

## 3. GAN 훈련 안정화 시도 및 결과

### 관찰된 불안정 현상

그림 2(학습 곡선)에서 다음과 같은 불안정 패턴이 관찰되었다.

- **Adversarial loss spike**: step 500~700, 1500 구간에서 adversarial loss(초록)가 60~130 수준으로 폭발적으로 증가
- **D loss spike**: 동일 구간에서 Discriminator loss(주황)가 20~35까지 급등
- 이는 D와 G 사이의 균형이 일시적으로 붕괴하여 G가 D를 전혀 속이지 못하는 상태가 반복됨을 의미한다.

### 적용한 안정화 전략

| 전략 | 설정값 | 근거 |
|---|---|---|
| 2-Phase 훈련 | Phase 1: MSE 1000 스텝 사전훈련 후 Phase 2 진입 | G가 기본 SR 능력 없이 GAN 훈련 시작 시 mode collapse 위험 |
| 낮은 adversarial 가중치 | $\lambda_{adv}=0.001$ | G loss에서 adversarial 항의 과도한 지배 방지 |
| 낮은 perceptual 가중치 | $\lambda_{perc}=0.006$ | pixel loss 대비 스케일 조정 |
| D/G 업데이트 비율 | D : G = 1 : 1 | D가 너무 강해지지 않도록 동일 빈도 유지 |
| Adam optimizer | $\beta_1=0.9$, lr=$10^{-4}$ | GAN 논문 권장 설정 |

### 결과

G total loss는 초반 0.10에서 최종적으로 0.02~0.03 수준으로 감소하였으며, spike 구간 이후에도 전반적인 하강 추세를 유지하였다. 완전한 안정화에는 이르지 못했으나 훈련이 발산하지 않고 수렴하는 경향을 보였다.

---

## 4. 하이퍼파라미터 선택 근거 및 조정 과정

| 하이퍼파라미터 | 최종값 | 선택 근거 |
|---|---|---|
| SCALE | 4 | 데이터 LR/HR 크기 비율에서 자동 감지 |
| HR_SIZE | 256 | 512×512 사용 시 GPU OOM 발생 → 256으로 축소 |
| BATCH_SIZE | 4 | 256×256 전체 이미지 기준 T4 GPU 수용 가능한 최대값 |
| NUM_RES_BLOCKS | 16 | 원 논문(Ledig et al., 2017) 권장값 |
| PRETRAIN_STEPS | 1000 | MSE loss가 수렴 안정화되는 스텝 수 확인 후 결정 |
| SRGAN_STEPS | 2000 | G total loss 하강 추세 확인 후 결정 |
| $\lambda_{pixel}$ | 1.0 | 기준 가중치 |
| $\lambda_{perc}$ | 0.006 | VGG feature map 값의 스케일이 픽셀 값보다 크므로 낮게 설정 |
| $\lambda_{adv}$ | 0.001 | GAN 훈련 초기 안정성 확보를 위해 최소화 |
| LR_G, LR_D | $10^{-4}$ | SRGAN 논문 권장값 |

---

## 5. 결과 요약

### 정량 평가

| 방법 | PSNR (dB) | SSIM |
|---|---|---|
| LR Bicubic | ~22.5 | ~0.34 |
| SR (SRGAN) | ~22.9 | ~0.31 |

PSNR은 SRGAN이 Bicubic 대비 소폭 향상되었으나 SSIM은 일부 낮아졌다. 이는 SRGAN의 특성상 perceptual quality를 위해 pixel-level 정확도(PSNR/SSIM)를 일부 희생하는 trade-off가 발생하기 때문이다. 실제 영상을 보면 SR 결과는 Bicubic 대비 노이즈가 억제되고 망막층 경계가 더 선명하게 표현됨을 확인할 수 있다.

### 결론

Pixel loss만 사용한 Phase 1 결과와 비교하여, Perceptual loss 및 Adversarial loss를 추가한 SRGAN(Phase 2)은 정량 지표(PSNR)의 큰 향상보다 정성적 선명도 개선에 기여하였다. 이는 SRGAN이 PSNR 최적화보다 perceptually 더 선명한 영상 생성을 목표로 설계된 모델임을 실험적으로 확인한 결과이다.

---

**참고문헌**  
Ledig, C., et al. "Photo-realistic single image super-resolution using a generative adversarial network." CVPR 2017.

# 논문 골격: MNIST FGSM 적대적 훈련과 전이성 분석

## 1. 서론

- MNIST 분류 모델을 대상으로 적대적 예제 취약성과 방어 성능을 재현 가능하게 분석한다.
- 단순 FGSM 재현을 넘어 두 모델 구조와 두 학습 방법을 교차 비교한다.
- 방어 모델의 white-box 강건성과 모델 간 전이 공격 강건성을 구분한다.

## 2. 연구 배경

### 적대적 예제

- 사람이 보기에는 원본과 유사하지만 모델의 예측을 바꾸는 입력 교란이다.
- 본 연구는 `[0, 1]` 픽셀 공간에서 L-infinity 제약을 사용한다.

### FGSM

- 손실 함수의 입력 gradient 부호를 이용하는 1-step untargeted 공격이다.
- 수식: `x_adv = clamp(x + epsilon * sign(gradient_x loss(model(x), y)), 0, 1)`.

### adversarial training

- 학습 중 현재 모델이 생성한 적대적 예제를 함께 사용해 강건성을 높이는 방법이다.
- 본 연구에서는 clean loss와 adversarial loss를 각각 `0.5`로 가중한다.

### 군 AI 적용 동기

- 군 AI 시스템은 환경 변화, 센서 잡음, 의도적 교란에 대한 안정성이 중요하다.
- 본 연구는 MNIST라는 제한된 실험 환경에서 강건성 평가 절차와 지표를 명확히 하는 데 초점을 둔다.

## 3. 연구 질문

“FGSM 적대적 훈련으로 향상된 강건성이 다른 구조의 모델에서 생성된 전이 공격과 더 강한 반복 공격인 PGD에도 유지되는가?”

## 4. 연구 방법

- 데이터셋: MNIST.
- 모델: LeNet, SmallCNN.
- 학습 방식: standard training, FGSM adversarial training.
- 최종 모델: `lenet_standard`, `smallcnn_standard`, `lenet_fgsm_at`, `smallcnn_fgsm_at`.
- 공격: FGSM, PGD L-infinity.

## 5. 실험 환경

| 항목 | 값 |
|---|---|
| Python | TODO: `results/environment.json`에서 기록 |
| PyTorch | TODO: `results/environment.json`에서 기록 |
| TorchVision | TODO: `results/environment.json`에서 기록 |
| 장치 | TODO: `results/environment.json`에서 기록 |
| seed | TODO: `results/run_config.json`에서 기록 |

## 6. 평가 지표

- Clean accuracy.
- Robust accuracy.
- Attack success rate.
- Conditional transfer success rate.
- Clean accuracy retention.
- 군 적용 참고 자료에서 제시한 작전운용성능 참고 목표치 90%와 epsilon `0.25` robust accuracy 비교.

## 7. 실험 결과

| 모델 | Clean accuracy | FGSM epsilon=0.25 robust accuracy | Clean accuracy retention |
|---|---:|---:|---:|
| lenet_standard | TODO | TODO | TODO |
| smallcnn_standard | TODO | TODO | TODO |
| lenet_fgsm_at | TODO | TODO | TODO |
| smallcnn_fgsm_at | TODO | TODO | TODO |

실행하지 않은 결과에는 임의 숫자를 넣지 않는다.

## 8. 전이성 분석

- 행은 source model, 열은 target model이다.
- 순서: `lenet_standard`, `smallcnn_standard`, `lenet_fgsm_at`, `smallcnn_fgsm_at`.
- 대각선은 white-box FGSM, 비대각선은 transfer attack이다.

| Source / Target | lenet_standard | smallcnn_standard | lenet_fgsm_at | smallcnn_fgsm_at |
|---|---:|---:|---:|---:|
| lenet_standard | TODO | TODO | TODO | TODO |
| smallcnn_standard | TODO | TODO | TODO | TODO |
| lenet_fgsm_at | TODO | TODO | TODO | TODO |
| smallcnn_fgsm_at | TODO | TODO | TODO | TODO |

## 9. 논의

- FGSM 적대적 훈련이 white-box FGSM에는 강건성을 높이는지 분석한다.
- 동일 방어가 전이 공격에도 유지되는지 source-target 방향성을 구분해 해석한다.
- FGSM 적대적 훈련이 PGD에도 일반화되는지 추가 검증한다.
- 정상 정확도와 강건 정확도의 trade-off를 정량화한다.

## 10. 군 적용 시사점

- 작전 환경의 AI는 clean 성능뿐 아니라 입력 교란 하 성능을 함께 보고해야 한다.
- 본 실험의 90% 값은 군 적용 참고 자료에서 제시한 작전운용성능 참고 목표치로만 사용하며, 공식적·보편적 기준으로 단정하지 않는다.
- MNIST 결과를 실제 군사 자산 이미지나 물리 환경으로 직접 일반화하지 않는다.

## 11. 한계

- MNIST는 실제 시각 인식 환경보다 단순하다.
- FGSM와 PGD만 사용한다.
- 탐지, 정화, 물리 패치, 객체탐지는 다루지 않는다.
- quick 모드 결과는 성능 결론에 사용할 수 없다.

## 12. 향후 연구

- CIFAR-10 확장.
- 직접 촬영 데이터.
- 군사 자산 이미지.
- 물리적 적대적 패치.
- 객체탐지 모델.
- BIM, DeepFool 등 추가 공격.
- 탐지 및 정화 방어.
- 새로운 공격 알고리즘 개발.

## 13. 결론

- 본 연구는 FGSM adversarial training을 두 CNN 구조와 네 모델 조합에서 재현 가능하게 비교한다.
- white-box 강건성, 모델 간 전이성, PGD 보조 평가를 분리해 방어 효과의 범위를 점검한다.
- 최종 결론은 실제 실행 결과 CSV와 자동 생성된 `results/summary.md`에 근거해 작성한다.

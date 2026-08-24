# CNN 진행사항 및 향후 계획

## 1. 현재까지 진행한 사항

* PyTorch 기반 1D CNN 모델 구현
* EEG 데이터를 128행 Window 단위로 구성하여 모델 입력으로 사용
* 19개 EEG 채널을 이용하여 `(19, 128)` 형태의 데이터를 입력
* Train / Validation 데이터를 이용한 학습 및 검증 진행
* Epoch별 Train Loss, Validation Loss, Validation Accuracy 확인
* Validation Accuracy를 기준으로 Best Epoch 기록

### 1.1 과적합 확인

CNN Baseline 모델의 Train Loss와 Validation Loss를 비교하여 과적합 여부를 확인하였다.

### 1.2 Dropout 적용

* `Dropout(p=0.5)` 적용
* 일부 뉴런을 학습 과정에서 제외하여 특정 학습 데이터에 지나치게 의존하는 것을 방지
* Baseline과 Validation 성능 비교

### 1.3 Weight Decay 적용

* `weight_decay=0.0001` 적용
* 모델의 가중치가 지나치게 커지는 것을 억제하여 과적합 감소 여부 확인
* Dropout과 별도로 적용하여 성능 비교

### 1.4 Dropout + Weight Decay 적용

* `Dropout(p=0.5)`
* `weight_decay=0.0001`
* 두 과적합 방지 방법을 함께 적용하여 성능 변화 확인
* 현재 실험에서는 과적합이 충분히 개선되지 않아 추가적인 방법 검토 필요

---

## 2. 앞으로 진행할 사항

### 2.1 Early Stopping 적용

* Validation 성능이 일정 Epoch 동안 개선되지 않을 경우 학습을 중단
* 모델이 학습 데이터를 지나치게 학습하기 전에 학습을 종료하여 과적합 방지
* Best Epoch의 모델을 저장하여 최종 평가에 사용

### 2.2 CNN 모델 구조 단순화

Early Stopping 적용 후에도 과적합이 지속될 경우 모델의 복잡도를 낮추는 방법을 검토한다.

검토 항목:

* Conv1d Filter 수 감소
* Linear Layer의 뉴런 수 감소
* 필요할 경우 Layer 수 감소

모델이 데이터를 지나치게 외울 수 있는 능력을 줄여 일반화 성능 향상을 확인한다.

### 2.3 Data Augmentation 적용 검토

모델 구조를 조정한 후에도 과적합이 지속될 경우 EEG 데이터 증강을 검토한다.

* 기존 EEG 데이터를 의미가 훼손되지 않는 범위에서 변형
* 학습 데이터의 다양성을 증가
* 모델이 특정 학습 데이터의 패턴만 암기하는 현상을 감소시키는 것을 목표로 함

### 2.4 최종 성능 평가

과적합 감소 방법 적용 후 가장 성능이 좋은 CNN 모델을 선정하고 다음 평가 지표를 확인한다.

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix
* Train Loss
* Validation Loss

### 2.5 기존 머신러닝 모델과 비교

최종 CNN 모델의 성능을 기존 특징 추출 기반 머신러닝 모델과 비교하여 성능 차이를 확인한다.

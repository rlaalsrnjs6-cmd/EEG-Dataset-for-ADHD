# 1. 실험 결과

- cnn모델 학습

| 모델 | 단계 | Feature | epoch | Best Epoch | Val Accuracy | Train Loss | Val Loss | datail |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| eeg_cnn | cnn_baseline | Window(19,128) batch 32 | 10 | 5 | 0.7614 | 0.0777 | 2.2599 | 과적합 여부 확인을 위한 cnn_baseline 실험
| eeg_cnn | cnn_dropout_05 | Window(19,128) batch 32 | 10 | 7 | 0.7706 | 0.1126 | 3.0444 | Dropout(p=0.5) 적용 후 과적합 감소 여부 확인
| eeg_cnn | cnn_weight_decay_0001 | Window(19,128) batch 32 | 10 | 1 | 0.7577 | 1.8014 | 0.6846 | weight_decay=0.0001 단독 적용 후 과적합 감소 여부 확인
| eeg_cnn | cnn_dropout_05_weight_decay_0001 | Window(19,128) batch 32 | 10 | 7 | 0.1216 | 0.1216 | 2.9869 | Dropout(p=0.5) weight_decay=0.0001 적용 후 과적합 감소 여부 확인

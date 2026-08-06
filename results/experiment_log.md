목적
EEG 채널값을 사용한 기준 성능 확인

데이터 분할
id 단위로 train / validation / test 
동일 if가 다른 데이터 셋에 포함되지않도록 처리
adhd와 control을 분리후 70%, 15%, 15%로 분리

입력 데이터
EEG채널 19개
EEG 한 행을 하나의 학습 샘플로 사용
id 컬럼 제외
ADHD = 1, Control = 0

모델 
LogisticRegression

Validation 결과 
Accuracy : 0.6015
Precision : 0.6016
Recall : 0.9996
F1-score : 0.7511

분석
모델이 Validation 데이터 대부분을 ADHD로 예측

표준화가 진행되지 않은 점과 한사람이 아닌 한행마다 ADHD여부를 판단하게 학습시킨것 문제가 있어보임

다음실험
우선 표준화를 실행하고 결과에 변동이 없다면
연속된 EEG행을 구간으로 묶고 동일한 모델로 다시 학습

StandardScaler적용 후
데이터에 유의미한 변화가 업서어 128행을 묶어 동일한 모델로 학습

128행 구간 특징 추출
128행을 하나의 구간으로 묶었다
각 채널의 평균과 표준편차를 특징으로 사용
정확도가 약 0.6822로 상승
ADHD로만 예측하는 현상이 크게 감소
이후 min과 max를 추가해보았음
유의미한 변화 없음
다음 실험 
모델 변경 후 기록

baseline결론
한행마다 ADHD를 판결하니 대부분의 평가단계에서 ADHD만을 가르켯으나
128행단위로 묶어 학습하니 ADHD만을 예측하는 문제가 크게 개선

RandomForest로 모델 변경 후
정확도, 정밀도, 재현율, F1스코어가 모두 크게 감소

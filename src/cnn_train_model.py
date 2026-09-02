from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from utils import save_log

torch.manual_seed(42)

batch_size = 32

# npz파일 읽어오기
data = np.load("data/cnn_processed/cnn_data5.npz")
train_array = data["train_windows"]
train_labels = data["train_labels"]
validation_array = data["validation_windows"]
validation_labels = data["validation_labels"]
test_array = data["test_windows"]
test_labels = data["test_labels"]
real_array = data["real_windows"]
real_labels = data["real_labels"]

# 파이토치가 쓸수있는 텐서 형태로 변환
train_tensor = torch.tensor(
    train_array,
    dtype=torch.float32
)
train_labels_tensor = torch.tensor(
    train_labels,
    dtype=torch.long
)
validation_tensor = torch.tensor(
    validation_array,
    dtype=torch.float32
)
validation_labels_tensor = torch.tensor(
    validation_labels,
    dtype=torch.long
)
test_tensor = torch.tensor(
    test_array,
    dtype=torch.float32
)
test_labels_tensor = torch.tensor(
    test_labels,
    dtype=torch.long
)
real_tensor = torch.tensor(
    real_array,
    dtype=torch.float32
)
real_labels_tensor = torch.tensor(
    real_labels,
    dtype=torch.long
)

# 입력데이터와 라벨을 묶어둠
train_dataset = TensorDataset(
    train_tensor,
    train_labels_tensor
)
validation_dataset = TensorDataset(
    validation_tensor,
    validation_labels_tensor
)
test_dataset = TensorDataset(
    test_tensor,
    test_labels_tensor
)
real_dataset = TensorDataset(
    real_tensor,
    real_labels_tensor
)

# 배치 사이즈 설정 데이터를 얼마나 보여줄지
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle= True
)
validation_loader = DataLoader(
    validation_dataset,
    batch_size=batch_size,
    shuffle= False
)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle= False
)
real_loader = DataLoader(
    real_dataset,
    batch_size=batch_size,
    shuffle= False
)

# eeg_cnn 객체 생성
class eeg_cnn(nn.Module):
    def __init__(self):
        # nn.Module의 init 실행
        super().__init__()

        #1차원 합성곱 특징을 찾아냄
        self.conv1 = nn.Conv1d(
            # 19채널이기때문에 19
            in_channels=19,
            # 다른 패턴을 찾는 필터 개수
            out_channels=32,
            # 시간축에서 연속된 3개 시점씩 확인
            kernel_size=3
        )

        # 음수는 0으로, 양수는 그대로 유지하는 활성화 함수
        self.relu = nn.ReLU()

        # Conv1d의 데이터의 길이를 줄이는 역할
        # 2  7  3  5  1  8 을 [2,7] [3,5] [1,8] 로 묶어 큰  값을 선택 7 5 8
        # 데이터의 크기를 줄이고 강하게 나타난 특징을 남김
        self.pool = nn.MaxPool1d(
            kernel_size=2
        )

        self.conv2 = nn.Conv1d(
            in_channels=32,
            out_channels=64,
            kernel_size=3
        )

        # Linear에 넣기위해 채널과 시간축을 하나의 특징 벡터로 펼침
        self.flatten = nn.Flatten()

        # 활성값 일부를 랜덤하게 0으로 바꿈 과적합 방지
        self.dropout = nn.Dropout(
            p=0.5
        )

        # ADHD / Control 분류를 위한 최종 출력 생성
        self.fc = nn.Linear(
            #  현재로선 1920개의 특징
            in_features=10112,
            # 출력 뉴런
            out_features=2
        )

        # 입력 EEG 데이터가 각 신경망 Layer를 어떤 순서로 통과할지 정의
    def forward(self,x):
        x=self.conv1(x)
        x=self.relu(x)
        x=self.pool(x)

        x=self.conv2(x)
        x=self.relu(x)
        x=self.pool(x)

        x=self.flatten(x)
        x=self.dropout(x)
        x=self.fc(x)
        return x

# eeg_cnn 객체 생성
model = eeg_cnn()

# 예측과 정답을 비교해 로스계산
# 정답 클래스에 대한 모델의 점수가 낮을수록 큰 Loss를 부여
criterion = nn.CrossEntropyLoss()

# 로스를 줄이는 방향으로 가중치 수정
optimizer = torch.optim.Adam(
    model.parameters(),
    #가중치를 한 번에 얼마나 크게 수정할지
    lr=0.001,
    # 가중치가 지나치게 커지는 것을 억제 과적합 방지
    weight_decay=0.0001
)

#몇번 반복할지
epochs = 10;
#로그용
best_validation_accuracy = 0
best_epoch = 0
train_loss_at_best = 0
validation_loss_at_best = 0

for epoch in range(epochs):
    total_loss = 0
    validation_loss = 0
    correct = 0
    total = 0 
    # 모델을 학습용으로 변경
    # 드롭아웃 작동
    model.train()
    for batch_windows, batch_labels in train_loader:

        # 이전 배치에서 계산된 기울기 초기화
        optimizer.zero_grad()

        # forward순서대로 통과시켜 예측값 생성
        # 신경망 입력
        output = model(batch_windows)

        # 로스계산
        loss = criterion(output, batch_labels)

        # 로스기준 가중치 기울기 계산
        loss.backward()

        # 계산된 기울기를 이용해 실제 가중치 수정
        optimizer.step()

        # 현재 배치의 로스 값 누적
        total_loss += loss.item()

    # 모델을 검증용으로 변경
    # 드롭아웃 작동x
    model.eval()
    #학습이 아니기에 가중치 수정x 기울기 계산할 필요없다
    with torch.no_grad():
        for batch_windows, batch_labels in validation_loader:

            # 포워드 순서대로 예측값 생성
            output = model(batch_windows)

            # 로스 계산
            loss = criterion(output, batch_labels)

            validation_loss += loss.item()

            # 두 클래스의 출력값중 가장 큰 값의 인덱스를 최종 예측 라벨로 선택
            predicted = output.argmax(dim=1)

            correct += (predicted == batch_labels).sum().item()

            total += batch_labels.size(0)

    # 트레인 배치들의 평균 로스계산
    average_loss = total_loss / len(train_loader)
    # 벨리데이션 배치들의 평균 로스 계산
    average_validation_loss = validation_loss / len(validation_loader)
    # 벨리데이션 정확도 계산
    validation_accuracy = correct / total
    
    if validation_accuracy > best_validation_accuracy:
        best_validation_accuracy = validation_accuracy
        best_epoch = epoch + 1
        train_loss_at_best = average_loss
        validation_loss_at_best = average_validation_loss
    print("epoch",epoch+1)
    print("average_loss",average_loss)
    print("average_validation_loss",average_validation_loss)
    print("validation_accuracy",validation_accuracy)
print("=====================================================")
print("best_validation_accuracy", best_validation_accuracy)
print("best_epoch", best_epoch)
print("train_loss_at_best", train_loss_at_best)
print("bestvalidation_loss_at_best_epoch", validation_loss_at_best)

cnn_path = Path("logs/cnn_train_logs.csv")

cnn_record = {
    "run_time": datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S"),

    "experiment": "cnn_dropout_05_weight_decay_0001",
    "model": "eeg_cnn",
    "learning_rate": 0.001,
    "batch_size": batch_size,
    "epochs": epochs,
    "random_seed": 42,

    "best_validation_accuracy": best_validation_accuracy,
    "best_epoch": best_epoch,
    "train_loss_at_best": train_loss_at_best,
    "validation_loss_at_best": validation_loss_at_best,

    "experiment_detail": "weight_decay=0.0001_drop=(p=0.5)",
    "note": "단독 weight_decay 적용후에도 과적합 확인 드롭아웃 웨이트디케이 동시적용 후 과적합 확인"
}
save_log(cnn_record,cnn_path)
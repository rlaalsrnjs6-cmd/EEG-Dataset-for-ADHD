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

data = np.load("data/cnn_processed/cnn_data.npz")
train_array = data["train_windows"]
train_labels = data["train_labels"]
validation_array = data["validation_windows"]
validation_labels = data["validation_labels"]
test_array = data["test_windows"]
test_labels = data["test_labels"]
real_array = data["real_windows"]
real_labels = data["real_labels"]

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

class eeg_cnn(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels=19,
            out_channels=32,
            kernel_size=3
        )

        self.relu = nn.ReLU()

        self.pool = nn.MaxPool1d(
            kernel_size=2
        )

        self.conv2 = nn.Conv1d(
            in_channels=32,
            out_channels=64,
            kernel_size=3
        )

        self.flatten = nn.Flatten()

        self.dropout = nn.Dropout(
            p=0.5
        )

        self.fc = nn.Linear(
            in_features=1920,
            out_features=2
        )

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

model = eeg_cnn()

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=0.0001
)

epochs = 10;
best_validation_accuracy = 0
best_epoch = 0
train_loss_at_best = 0
validation_loss_at_best = 0
for epoch in range(epochs):
    total_loss = 0
    validation_loss = 0
    correct = 0
    total = 0 
    model.train()
    for batch_windows, batch_labels in train_loader:

        optimizer.zero_grad()

        output = model(batch_windows)

        loss = criterion(output, batch_labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    model.eval()

    with torch.no_grad():
        for batch_windows, batch_labels in validation_loader:

            output = model(batch_windows)

            loss = criterion(output, batch_labels)

            validation_loss += loss.item()

            predicted = output.argmax(dim=1)

            correct += (predicted == batch_labels).sum().item()

            total += batch_labels.size(0)

    average_loss = total_loss / len(train_loader)
    average_validation_loss = validation_loss / len(validation_loader)
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

    "experiment": "cnn__drop_05_weight_decay_0001",
    "model": "eeg_cnn",
    "learning_rate": 0.001,
    "batch_size": batch_size,
    "epochs": epochs,
    "random_seed": 42,

    "best_validation_accuracy": best_validation_accuracy,
    "best_epoch": best_epoch,
    "train_loss_at_best": train_loss_at_best,
    "validation_loss_at_best": validation_loss_at_best,

    "experiment_detail": "드롭아웃 적적용 weight_decay=0.0001 적용",
    "note": "weight_decay 단독 적용 후에도 과적합 지속"
}
save_log(cnn_record,cnn_path)
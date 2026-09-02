import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

eeg_data = np.load("data/cnn_processed/cnn_data.npz")

train_tensor = torch.tensor(
        eeg_data["train_windows"],
        dtype=torch.float32
    )
train_labels_tensor = torch.tensor(
    eeg_data["train_labels"],
    dtype=torch.long
)
validation_tensor = torch.tensor(
        eeg_data["validation_windows"],
        dtype=torch.float32
    )
validation_labels_tensor = torch.tensor(
    eeg_data["validation_labels"],
    dtype=torch.long
)
test_tensor = torch.tensor(
        eeg_data["test_windows"],
        dtype=torch.float32
    )
test_labels_tensor = torch.tensor(
    eeg_data["test_labels"],
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

train_dataload = DataLoader(
    train_dataset,
    batch_size = 30,
    shuffle=True
)
validation_dataload = DataLoader(
    validation_dataset,
    batch_size = 30,
    shuffle=True
)
test_dataload = DataLoader(
    test_dataset,
    batch_size = 30,
    shuffle=True
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

        self.flatten = nn.Flatten()

        self.linear = nn.Linear(
            in_features=2016,
            out_features=2 
        )

    def forward(self,x):
        x=self.conv1(x)
        x=self.relu(x)
        x=self.pool(x)
        x=self.flatten(x)
        x=self.linear(x)
        return(x)
    
model = eeg_cnn()

entropy = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0001
)

epochs = 10

for epoch in range(epochs):
    validation_loss = 0
    correct = 0
    total = 0
    train_loss = 0
    train_correct = 0
    train_total = 0

    model.train()
    for batch_window, batch_label in train_dataload:
        optimizer.zero_grad()

        output = model(batch_window)

        loss = entropy(output,batch_label)

        train_loss += loss.item() * batch_label.size(0)

        predicted = output.argmax(dim=1)

        train_correct += (predicted == batch_label).sum().item()
        train_total += batch_label.size(0)

        loss.backward()

        optimizer.step()

    model.eval()
    with torch.no_grad():
        for batch_window, batch_label in validation_dataload:
            output = model(batch_window)

            predicted = output.argmax(dim=1)

            loss = entropy(output,batch_label)

            validation_loss += loss.item() * batch_label.size(0)

            correct += (predicted == batch_label).sum().item()
            total += batch_label.size(0)

    validation_loss = validation_loss / total
    validation_accuracy = correct / total
    train_loss = train_loss / train_total
    train_accuracy = train_correct / train_total
    print("epoch",epoch+1)
    print("train loss:", train_loss)
    print("train accuracy:", train_accuracy)
    print("validation loss:", validation_loss)
    print("validation accuracy:", validation_accuracy)
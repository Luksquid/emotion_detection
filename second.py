import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchmetrics import Precision, Recall

print(torch.cuda.is_available())

class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
train_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.Resize((64, 64)),
])

dataset_train = ImageFolder(
    "./train_samples",
    transform=train_transforms,    
)

net = Net(num_classes=3)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

dataloader_train = DataLoader(
    dataset_train,
    shuffle=True,
    batch_size=1,
)

num_epochs = 3

for epoch in range(num_epochs):
    print(f'Epoch [{epoch+1}/{num_epochs}]')
    for images, labels in dataloader_train:
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((64, 64)),
])

dataset_test = ImageFolder(
    "./test_samples",
    transform=test_transforms,
)

metric_precision = Precision(
    task="multiclass", num_classes=3, average="macro"
)

metric_recall = Recall(
    task="multiclass", num_classes=3, average="macro"
)

dataloader_test = DataLoader(
    dataset_test,
    shuffle=True,
    batch_size=1,
)

net.eval()
with torch.no_grad():
    for images, labels in dataloader_test:
        outputs = net(images)
        _, preds = torch.max(outputs, 1)
        metric_precision(preds, labels)
        metric_recall(preds, labels)
        
precision = metric_precision.compute()
recall = metric_recall.compute()

print(precision, recall)
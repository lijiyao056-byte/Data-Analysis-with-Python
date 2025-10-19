import torch
from torch import nn
from torch.nn import functional as F

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Определяем слои сети
        self.conv1 = nn.Conv2d(3, 3, kernel_size=(5, 5))  # 3 входных канала, 3 фильтра
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(3, 5, kernel_size=(3, 3))  # 3 входных канала, 5 фильтров
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        self.flatten = nn.Flatten()
        
        # Вычисляем размер после сверточных слоев
        # 32x32 -> conv1(5x5) -> 28x28 -> pool1(2x2) -> 14x14
        # 14x14 -> conv2(3x3) -> 12x12 -> pool2(2x2) -> 6x6
        # 5 фильтров * 6 * 6 = 180 элементов
        self.fc1 = nn.Linear(5 * 6 * 6, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        # размерность х ~ [64, 3, 32, 32]
        
        # Первый сверточный блок
        x = self.conv1(x)
        x = F.relu(x)  # Используем F.relu как требуется
        x = self.pool1(x)
        
        # Второй сверточный блок
        x = self.conv2(x)
        x = F.relu(x)  # Используем F.relu как требуется
        x = self.pool2(x)
        
        # Полносвязные слои
        x = self.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)  # Используем F.relu как требуется
        x = self.fc2(x)
        
        return x

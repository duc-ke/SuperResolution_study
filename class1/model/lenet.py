import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=10): # img size
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.act = nn.ReLU()
    
    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.pool(x)
        x = self.act(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)  # input size 바뀔때 유연하게 바꾸기
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.fc3(x)    # 마지막 act는 필요 없나

        return x

def test():
    img_size = 32
    img_channel = 3
    num_classes = 10
    batch_size = 5

    # sample 
    input = torch.rand(batch_size, img_channel, img_size, img_size)

    model = LeNet(in_channels=img_channel, num_classes=num_classes)
    output = model(input)

    print(f"model input shape: {input.shape}")
    print(f"model output shape: {output.shape}")

if __name__ == "__main__":
    test()


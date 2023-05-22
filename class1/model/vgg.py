import torch
import torch.nn as nn
import torch.nn.functional as F


class VGG19(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        self.conv1 = nn.Conv2d(self.in_channels, 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.b1 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, 1)
        self.b2 = nn.BatchNorm2d(128)
        
        self.conv5 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv6_8 = nn.Conv2d(256, 256, 3, 1, 1)
        self.b3 = nn.BatchNorm2d(256)
        
        self.conv9 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv10_16 = nn.Conv2d(512, 512, 3, 1, 1)
        self.b4 = nn.BatchNorm2d(512)
        
        self.fc1 = nn.Linear(512*1*1, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, self.num_classes)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.act = nn.ReLU()
        self.do = nn.Dropout()
        
        
    def forward(self, x):
        x = self.act(self.b1(self.conv1(x)))
        x = self.act(self.b1(self.conv2(x)))
        x = self.pool(x)
        
        x = self.act(self.b2(self.conv3(x)))
        x = self.act(self.b2(self.conv4(x)))
        x = self.pool(x)
        
        x = self.act(self.b3(self.conv5(x)))
        x = self.act(self.b3(self.conv6_8(x)))
        x = self.act(self.b3(self.conv6_8(x)))
        x = self.act(self.b3(self.conv6_8(x)))
        x = self.pool(x)
        
        x = self.act(self.b4(self.conv9(x)))
        x = self.act(self.b4(self.conv10_16(x)))
        x = self.act(self.b4(self.conv10_16(x)))
        x = self.act(self.b4(self.conv10_16(x)))
        x = self.pool(x)
        
        x = self.act(self.b4(self.conv10_16(x)))
        x = self.act(self.b4(self.conv10_16(x)))
        x = self.act(self.b4(self.conv10_16(x)))
        x = self.act(self.b4(self.conv10_16(x)))
        x = self.pool(x)

        x = x.view(-1, 512*1*1)
        
        x = self.do(self.act(self.fc1(x)))
        x = self.do(self.act(self.fc2(x)))
        x = self.do(self.act(self.fc3(x)))
        
        return x
        

# VGG type dict
# int : output chnnels after conv layer
# 'M' : max pooling layer
VGG_types = {
    'VGG11' : [64, 'M', 128, 'M', 256, 256, 'M', 512,512, 'M',512,512,'M'],
    'VGG13' : [64,64, 'M', 128, 128, 'M', 256, 256, 'M', 512,512, 'M', 512,512,'M'],
    'VGG16' : [64,64, 'M', 128, 128, 'M', 256, 256,256, 'M', 512,512,512, 'M',512,512,512,'M'],
    'VGG19' : [64,64, 'M', 128, 128, 'M', 256, 256,256,256, 'M', 512,512,512,512, 'M',512,512,512,512,'M']
}

class VGG(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, model_type='VGG19'):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.conv_layers = self.create_conv_layers(VGG_types[model_type])
        self.fcs = nn.Sequential(
            nn.Linear(512 * 1 * 1, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 512*1*1)
        x = self.fcs(x)
        return x
    
    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels
        
        for x in architecture:
            if type(x) == int:  # int means convlayer
                out_channels = x
                
                layers += [
                    nn.Conv2d(
                        in_channels=in_channels, out_channels=out_channels, 
                        kernel_size=(3,3), stride=(1,1), padding=(1,1)),
                    nn.BatchNorm2d(x),
                    nn.ReLU()
                ]
                
                in_channels = x
            elif x == 'M':
                layers += [nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))]
        
        return nn.Sequential(*layers)
    


def test_vgg1():
    img_size = 32 # 224  # 32
    img_channel = 3
    num_classes = 10
    batch_size = 5

    # sample 
    input = torch.rand(batch_size, img_channel, img_size, img_size)

    model = VGG19(in_channels=img_channel, num_classes=num_classes)
    output = model(input)

    print(f"model input shape: {input.shape}")
    print(f"model output shape: {output.shape}")


def test_vgg2():
    img_size = 32 # 224  # 32
    img_channel = 3
    num_classes = 10
    batch_size = 5

    # sample 
    input = torch.rand(batch_size, img_channel, img_size, img_size)

    model = VGG(model_type="VGG19", in_channels=img_channel, num_classes=num_classes)
    output = model(input)

    print(f"model input shape: {input.shape}")
    print(f"model output shape: {output.shape}")

if __name__ == "__main__":
    test_vgg1()
    # test_vgg2()

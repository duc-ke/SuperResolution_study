import torch
import torchvision
from data import cifar10_dataloader
from model import cnn
import torch.optim as optim
from trainer import train
from tester import evaluation


batch_size = 4
learning_rate = 0.001
momentum = 0.9
epochs = 2
out_model_name = "./cifar_net.pth"


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

trainloader, testloader, classes = cifar10_dataloader.get_data(batch_size)

net = cnn.Net()
net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

train(net, epochs, trainloader, optimizer, criterion, device, modelname=out_model_name)


## test (Acc 성능확인)
net = cnn.Net()
net.load_state_dict(torch.load(out_model_name))

evaluation(net, testloader)


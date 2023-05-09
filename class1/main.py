import torch
import torch.nn as nn
from data import cifar10_dataloader
from model import cnn
import torch.optim as optim
from trainer import train
from tester import evaluation


batch_size = 4
learning_rate = 0.001
momentum = 0.9
epochs = 2
out_weight_fname = "./cifar_net.pth"
mode = 'test'  # train or test


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

trainloader, testloader, classes = cifar10_dataloader.get_data(batch_size)

net = cnn.Net()

if mode == "train":
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

    train(net, epochs, trainloader, optimizer, criterion, device, modelname=out_weight_fname)
elif mode == "test":
    ## test (Acc 성능확인)
    evaluation(net, out_weight_fname, testloader, classes, device)


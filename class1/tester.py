import torch

def evaluation(net, out_weight_fname, testloader, classes, device):
    net.load_state_dict(torch.load(out_weight_fname))
    net.to(device)

    dataiter = iter(testloader)
    images, labels = next(dataiter)
    print(images.shape, labels.shape)
    images, labels = images.to(device), labels.to(device)
    
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)
    print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                              for j in range(4)))
    
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            # images, labels = data
            images, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
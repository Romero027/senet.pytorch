import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
from senet.baseline import resnet20
from senet.se_resnet import se_resnet20


def main():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Assuming that we are on a CUDA machine, this should print a CUDA device:

    print(device)

    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                            shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
   
    if args.baseline:
        model = resnet20().to(device)
    else:
        model = se_resnet20(num_classes=10, reduction=args.reduction).to(device)

    model.load_state_dict(torch.load('checkpoint/cifar_senet.pth'))
    # for name, value in model.named_parameters():
    #     # value.requires_grad = False
    #     print('name: {0},\t grad: {1}'.format(name, value.requires_grad))
    torch.save(model, 'checkpoint/cifar_senet_full.pth')


    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # for epoch in range(20):  # loop over the dataset multiple times
    #     print(f"Start training epoch {epoch}")
    #     running_loss = 0.0
    #     print(len(trainloader))
    #     for i, data in enumerate(trainloader, 0):
    #         # get the inputs; data is a list of [inputs, labels]
    #         inputs, labels = data[0].to(device), data[1].to(device)

    #         # zero the parameter gradients
    #         optimizer.zero_grad()

    #         # forward + backward + optimize
    #         outputs = model(inputs)
    #         loss = criterion(outputs, labels)
    #         loss.backward()
    #         optimizer.step()

    #         # print statistics
    #         running_loss += loss.item()
    #         if i % 2000 == 1999:    # print every 2000 mini-batches
    #             print('[%d, %5d] loss: %.3f' %
    #                 (epoch + 1, i + 1, running_loss / 2000))
    #             running_loss = 0.0
            
    #         correct = 0
    #         total = 0
    #     with torch.no_grad():
    #         for data in testloader:
    #             images, labels = data[0].to(device), data[1].to(device)
    #             outputs = model(images)
    #             _, predicted = torch.max(outputs.data, 1)
    #             total += labels.size(0)
    #             correct += (predicted == labels).sum().item()

    #     print('Accuracy of the network on the 10000 test images: %d %%' % (
    #         100 * correct / total))



    # print('Finished Training')
    # PATH = './cifar_net.pth'
    # torch.save(model.state_dict(), PATH)

    # correct = 0
    # total = 0
    # with torch.no_grad():
    #     for data in testloader:
    #         images, labels = data[0].to(device), data[1].to(device)
    #         outputs = model(images)
    #         _, predicted = torch.max(outputs.data, 1)
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum().item()

    # print('Accuracy of the network on the 10000 test images: %d %%' % (
    #     100 * correct / total))


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--reduction", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--baseline", action="store_true")
    args = p.parse_args()
    main()

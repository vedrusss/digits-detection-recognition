__author__ = "Alexey Antonenko, vedrusss@gmail.com"

import os
import torch
from torchvision import datasets, transforms
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5,self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 13)
    
    def forward(self,x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 16*4*4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_net(dataloader, device):
    train_loader = dataloader
    # Instantiate a network
    net = LeNet5()
    net.to(device)

    # Define loss function and optimizer
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        net.parameters(),
        lr=0.001,
        momentum=0.9
    )

    # Training model
    loss_list = []
    for epoch in range(1000):
        running_loss = 0.0
        for batch_idx, data in enumerate(train_loader, start=0):
            images, labels = data                       # Read the data of a batch
            images.to(device)
            labels.to(device)
            optimizer.zero_grad()                       # Clear the gradient, initialize
            outputs = net(images)                       # Forward propagation
            loss = loss_function(outputs, labels)       # Calculation error
            loss.backward()                             # Backpropagation
            optimizer.step()                            # Weight update
            running_loss += loss.to('cpu').item()                 # Error accumulation
            # Print the loss value every 10 batches
            if batch_idx % 10 == 0:
                print('epoch:{} batch_idx:{} loss:{}'
                      .format(epoch+1, batch_idx+1, running_loss/300))
                loss_list.append(running_loss/10)
                running_loss = 0.0                  #Error clearing

    print('Finished Training.')

    # Print loss value change curve
    plt.plot(loss_list)
    plt.title('traning loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.show()
    return net


def test_net(net, dataloader, device):
    test_loader = dataloader

    correct = 0  # Predict the correct number
    total = 0    # Total number of pictures

    for data in test_loader:
        images, labels = data
        images.to(device)
        labels.to(device)
        outputs = net(images).to('cpu')
        _, predict = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predict == labels).sum()
    print('Test set accuracy rate {}%'.format(100*correct // total))

def demo_net():
    # Test your own handwritten numbers manually designed
    from PIL import Image
    I = Image.open('8.jpg')
    L = I.convert('L')
    plt.imshow(L, cmap='gray')

    transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1037,), (0.3081,))
    ])
    
    im = transform(L)  # [C, H, W]
    im = torch.unsqueeze(im, dim=0)  # [N, C, H, W]

    with torch.no_grad():
        outputs = net(im)
        _, predict = torch.max(outputs.data, 1)
        print(predict)


if __name__ == '__main__':
    data_dir = "/data/tasks/ocr_pipeline/calculator_font/digits_and_signs"

    input_size = 28
    batch_size = 64

    data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.Grayscale(1),
        #transforms.RandomResizedCrop(input_size),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.1037,), (0.3081,))
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.Grayscale(1),
        transforms.ToTensor(),
        transforms.Normalize((0.1037,), (0.3081,))
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    }
    
    # Create training and validation datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    # Create training and validation dataloaders
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}

    # Detect if we have a GPU available
    device = 'cpu' # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = train_net(dataloaders_dict['train'], device)
    # save the model
    save_path = f'digits_lenet5.pth'
    torch.save(net.state_dict(), save_path)
    print(f"Model has been trained to {save_path}")
    test_net(net, dataloaders_dict['val'], device)
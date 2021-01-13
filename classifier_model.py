__author__ = "Alexey Antonenko, vedrusss@gmail.com"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from PIL import Image
import json


class LeNet5(nn.Module):
    def __init__(self, num_classes):
        super(LeNet5,self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self._device = 'cpu'  #  default host
        self._idx_to_label = None
    
    def forward(self,x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 16*4*4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def set_device(self, device):
        self._device = device
        return self

    @property
    def input_size(self):
        return 28
    
    @property
    def data_transforms(self):
        # Data augmentation and normalization for training
        # Just normalization for validation
        return {
            'train': transforms.Compose([
                        transforms.Resize(self.input_size),
                        transforms.CenterCrop(self.input_size),
                        transforms.Grayscale(1),
                        transforms.RandomChoice([
                            transforms.RandomRotation(5),
                            #transforms.RandomErasing(),
                            transforms.RandomResizedCrop(self.input_size)
                        ]),
                        transforms.ToTensor(),
                        transforms.Normalize((0.1037,), (0.3081,))
                        #transforms.Normalize((128.0,), (128.0,))
                    ]),
            'val': transforms.Compose([
                        transforms.Resize(self.input_size),
                        transforms.CenterCrop(self.input_size),
                        transforms.Grayscale(1),
                        transforms.ToTensor(),
                        transforms.Normalize((0.1037,), (0.3081,))
                        #transforms.Normalize((128.0,), (128.0,))
                    ]),
            }

    @property
    def model_name(self):
        return 'lenet'

    @property
    def device(self):
        return self._device

    @property
    def idx_to_label(self):
        return self._idx_to_label

def train(model, dataloaders, criterion, optimizer, num_epochs):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(model.device)
                labels = labels.to(model.device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


####  API  ####

def initialize_cnn_model(num_labels=None, model_path=None):
    # Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
    model_name = "lenet" # "squeezenet"
    # Number of classes in the dataset
    idx_to_class = {int(k):v for k, v in json.load(open(get_labels_path(model_path))).items()} if model_path else None
    num_classes = num_labels if num_labels else len(idx_to_class)
    assert(num_classes is not None),"Provide num classes or pre-trained model"
    # LeNet5 
    model_ft = LeNet5(num_classes)
    num_ftrs = model_ft.fc3.in_features
    input_size = model_ft.input_size

    if model_path:
        model_ft.load_state_dict(torch.load(model_path))
        model_ft._idx_to_label = {int(k):v for k,v in json.load(open(model_path.replace('.pth', '_labels.json'))).items()}
        #print(model_ft.idx_to_label)

    # Print the model we just instantiated
    #print(model_ft)

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Send the model to GPU
    model_ft = model_ft.to(device).set_device(device)

    return model_ft, idx_to_class

def test_cnn_model(model_ft, cv_images, verbose=False):
    image_transformer = model_ft.data_transforms['val']
    model_ft.eval()
    results = []
    for cv_img in cv_images:
        im_pil = Image.fromarray(cv_img)
        im_t = image_transformer(im_pil)
        batch_t = torch.unsqueeze(im_t, 0)
        batch_t = batch_t.to(model_ft.device)
        out = model_ft(batch_t).to('cpu').detach().numpy()[0]
        label_ind = out.argmax()
        label_score = out[label_ind]
        label = model_ft.idx_to_label[label_ind]
        results.append(label_ind)
    return results

def train_cnn_model(model_ft, train_data_root, val_data_root):
    print(f"Running CNN training with train: {train_data_root} and val: {val_data_root}")
    # Batch size for training (change depending on how much memory you have)
    batch_size = 12
    # Number of epochs to train for
    num_epochs = 1000
    lr = 0.001
    # Flag for feature extracting. When False, we finetune the whole model,
    #   when True we only update the reshaped layer params
    feature_extract = False # True
    print("Initializing Datasets and Dataloaders...")
    # Create training and validation datasets
    image_datasets = {x: datasets.ImageFolder(data_root, model_ft.data_transforms[x]) for (data_root, x) in \
                      [(train_data_root, 'train'), (val_data_root, 'val')]}
    print("Dataset initiated")
    idx_to_class = {v:k for k,v in image_datasets['train'].class_to_idx.items()}
    print(idx_to_class)
    model_ft._idx_to_label = idx_to_class

    # Create training and validation dataloaders
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], 
                        batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}
    print("Dataloaders created")
    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t",name)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(params_to_update, lr=lr, momentum=0.9)

    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()

    # Train and evaluate
    model_ft, hist = train(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs)
    print("Model has been trained")
    return idx_to_class

def save_cnn_model(model_ft, idx_to_class, save_path):
    # save the model
    save_label_mapping_path = get_labels_path(save_path)
    torch.save(model_ft.state_dict(), save_path)
    json.dump(idx_to_class, open(save_label_mapping_path, 'w'))
    print(f"Model has been saved to {save_path}, {save_label_mapping_path}")

def get_labels_path(model_path):
    if model_path.endswith('.pth'):
        return model_path.replace('.pth', '_labels.json')
    else:
        return model_path + '_labels.json'

if __name__ == "__main__":
    # Top level data directory. Here we assume the format of the directory conforms
    #   to the ImageFolder structure
    data_dir = "/data/tasks/ocr_pipeline/calculator_font/digits_and_signs/digits_only/012345emtpy"
    train_data_dir = os.path.join(data_dir, 'train')
    val_data_dir = os.path.join(data_dir, 'test')

    model = initialize_cnn_model(len(os.listdir(train_data_dir)))
    idx_to_class_map = train_cnn_model(model, train_data_dir, val_data_dir)

    model_path = 'digits_lenet5.pth'
    save_cnn_model(model, idx_to_class_map, model_path)
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import torchvision.transforms as T
from Preprocessing import DataPreprocess
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

def train_model(model, trainx, trainy, testx, testy, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    losses = []
    train_accuracies = []
    val_accuracies = []
    test_accuracies = []
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
            for i in range(trainy.shape[0]):
                x = trainx[i, :, :, :]
                y = torch.LongTensor([trainy[i]])
                # print(y)
                transform = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                x = transform(x)
                x = x.unsqueeze(0)
                x = x.to(device)
                y = y.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(x)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, y)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * x.size(0)
                running_corrects += torch.sum(preds == y.data)
            # if phase == 'train':
                # scheduler.step()

            epoch_loss = running_loss / trainy.shape[0]
            epoch_acc = running_corrects.double() / trainy.shape[0]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            if phase == 'train':
                losses.append(epoch_loss)
                train_accuracies.append(round(epoch_acc.item(), 4))
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            # testing on test set
            if phase == 'val':
                test_acc = predict(model, testx, testy)
                test_accuracies.append(round(test_acc, 4))
                val_accuracies.append(round(epoch_acc.item(), 4))
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best Val Acc: {:4f}'.format(best_acc))
    print("Training losses : " + str(losses))
    print("Training accuracies : " + str(train_accuracies))
    print("Validation accuracies : " + str(val_accuracies))
    print("Test accuracies : " + str(test_accuracies))
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def predict(model, testx, testy):
    print('Checking accuracy on test set')
    num_correct = 0
    num_samples = 0
    # set model to evaluation mode
    with torch.no_grad():
        # model.eval()
        for i in range(testy.shape[0]):
            x = testx[i, :, :, :]
            y = torch.LongTensor([testy[i]])
            # print(y)
            transform = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            x = transform(x)
            x = x.unsqueeze(0)
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            preds = torch.max(scores, axis=1)
            num_correct += (preds[1] == y).sum()
            num_samples += 1
        # num_samples = y.cpu().detach().numpy().shape[1]
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
        return acc

model_ft = models.resnet18(pretrained=True)
feature_extracting = True
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
set_parameter_requires_grad(model_ft, feature_extracting)
num_ftrs = model_ft.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
# model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, 256),nn.ReLU(),nn.Linear(256,128),nn.ReLU(),nn.Linear(128,5))
model_ft.fc = nn.Linear(num_ftrs, 5)
# model_ft.fc2 = nn.Linear(128, 5)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.0001, momentum=0.9)
# optimizer_ft = optim.Adam(model_ft.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
# Decay LR by a factor of 0.1 every 7 epochs
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
exp_lr_scheduler = None
file_path = "Data"
dp = DataPreprocess()
# print(model_ft)
params_to_update = model_ft.parameters()
print("Params to learn:")
if feature_extracting:
    params_to_update = []
    for name, param in model_ft.named_parameters():
        if "4.1" in name:
            param.requires_grad = True
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t", name)
else:
    for name, param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t", name)


trainx, testx, trainy, testy = dp.prepare(file_path, 0.20)
model_ft = train_model(model_ft, trainx, trainy, testx, testy, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=20)
predict(model_ft, testx, testy)
torch.save(model_ft, "D:/UMass/CS 670/Project/Code/model_transfer_report.pth")
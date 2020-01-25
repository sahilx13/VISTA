
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from Preprocessing import DataPreprocess
import copy

USE_GPU = True
dtype = torch.float32  # we will be using float throughout this tutorial
if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
# Constant to control how frequently we print train loss
print_every = 100
print('using device:', device)


def flatten(x):
    N = x.shape[0]  # read in N, C, H, W
    return x.view(N, -1)


class Flatten(nn.Module):
    def forward(self, x):
        return flatten(x)


def train(trainx, trainy, testx, testy, model, optimizer, epochs=1):
    """
    Train a model on CIFAR-10 using the PyTorch Module API.
    Inputs:
    - model: A PyTorch Module giving the model to train.
    - optimizer: An Optimizer object we will use to train the model
    - epochs: (Optional) A Python integer giving the number of epochs to train for
    Returns: Nothing, but prints model accuracies during training.
    """
    # x = torch.from_numpy(x)
    # y = torch.from_numpy(y)
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    # train_loader = torch.utils.data.DataLoader(dataset=TensorDataset(x, y),
    #                                            batch_size=20, shuffle=True)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    losses = []
    test_accuracies = []
    train_accuracies = []
    for e in range(epochs):
        print("Epoch : " + str(e+1))
        epoch_loss = 0
        for i in range(trainy.shape[0]):
            # img = Image.fromarray(x)
            model.train()  # put model to training mode
            x = trainx[i, :, :, :]
            y = torch.LongTensor([trainy[i]])
            transform = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            x = transform(x)
            x = x.unsqueeze(0)
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            loss = F.cross_entropy(scores, y)
            epoch_loss = epoch_loss + loss
            optimizer.zero_grad()
            # This is the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            loss.backward()
            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()
            #model.zero_grad()
        train_loss = epoch_loss/trainy.shape[0]
        train_loss = round(train_loss.item(), 4)
        # training loss
        print("Training loss : " + str(train_loss))
        losses.append(train_loss)
        # train accuracy
        train_acc = check_accuracy(trainx, trainy, model)
        train_accuracies.append(train_acc)
        print("Training Accuracy : " + str(train_acc))
        # test accuracy
        test_acc = check_accuracy(testx, testy, model)
        test_accuracies.append(test_acc)
        print("Test Accuracy : " + str(test_acc))
        if test_acc > best_acc:
            best_acc = test_acc
            best_model_wts = copy.deepcopy(model.state_dict())
    print("Training losses : " + str(losses))
    print("Test accuracies : " + str(test_accuracies))
    print("Train accuracies : " + str(train_accuracies))
    # load best model weights
    torch.save(model, "D:/UMass/CS 670/Project/Code/model_last.pth")
    model.load_state_dict(best_model_wts)
    torch.save(model, "D:/UMass/CS 670/Project/Code/model_best.pth")



def check_accuracy(testx, testy, model):

    print('Checking accuracy on test set')
    num_correct = 0
    num_samples = 0
      # set model to evaluation mode
    with torch.no_grad():
        model.eval()
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
        acc = round(acc, 4)
        return acc

model = nn.Sequential(nn.Conv2d(3, 16, kernel_size=5, padding=1,stride = 1, bias=True), nn.ReLU(),
                      nn.Conv2d(16, 32, kernel_size=5, padding=1,stride=1, bias=True), nn.ReLU(),
                      nn.BatchNorm2d(32),
                      nn.MaxPool2d(kernel_size=2, stride=2),
                      nn.Conv2d(32, 32, kernel_size=5, padding=1, bias=True), nn.ReLU(),
                      nn.Conv2d(32,64 ,kernel_size=5, padding=1, bias=True), nn.ReLU(),
                      nn.BatchNorm2d(64),
                      nn.MaxPool2d(kernel_size=2, stride=2),
                      nn.Conv2d(64, 96, kernel_size=3, padding=1, bias=True), nn.ReLU(),
                      nn.Conv2d(96, 128, kernel_size=3, padding=1, bias=True), nn.ReLU(),
                      nn.BatchNorm2d(128),
                      nn.MaxPool2d(kernel_size=2, stride=2),

                       Flatten(), nn.Linear(128 * 26 * 26, 1024),
                       nn.ReLU(), nn.Dropout(0.3), nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.3),nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
                                                  nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 5))


optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
# You should get at least 70% accuracy
file_path = "Data"
dp = DataPreprocess()
trainx, testx, trainy, testy = dp.prepare(file_path, 0.15)
train(trainx, trainy, testx, testy, model, optimizer, epochs=15)





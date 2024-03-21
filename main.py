#Import basic libraries including torch and torchvision 
#Import image package and data package from torch

import torch
import torchvision
from torchvision import models
import torch.nn as nn 
from PIL import Image
import torch.utils.data as data
import torch.nn.functional as F
from tqdm import tqdm

#Creating transform for train dataset as well as validation and test
image_transform = torchvision.transforms.Compose([torchvision.transforms.Resize(255),
                               torchvision.transforms.CenterCrop(224),
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

transNoAugment = torchvision.transforms.Compose([torchvision.transforms.Resize(255), 
                                  torchvision.transforms.CenterCrop(224),
                                  torchvision.transforms.ToTensor(),
                                  torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#To get different transform in splitted data needed a new class for that
class MyLazyDataset(data.Dataset):
    def __init__(self, dataset, transforms=None):
        self.dataset = dataset
        self.transform = transforms

    def __getitem__(self, index):
        if self.transform:
            x = self.transform(self.dataset[index][0])
        else:
            x = self.dataset[index][0]
        y = self.dataset[index][1]
        return x, y
    
    def __len__(self):
        return len(self.dataset)

#Loading actual data from root directory using imageFolder module
placesSimpDataset = torchvision.datasets.ImageFolder(root = 'Places2_simp')

#Splitting dataset using random_split from data.random_split
train_placesSimpDataset, val_placesSimpDataset, test_placesSimpDataset = data.random_split(placesSimpDataset, [35000, 2500, 2500])

#Splitting dataset into train, val and test
train_dataset = MyLazyDataset(train_placesSimpDataset, transforms=image_transform)
val_dataset = MyLazyDataset(val_placesSimpDataset, transforms=image_transform)
test_dataset = MyLazyDataset(test_placesSimpDataset, transforms=image_transform)

#Now we need to cerate dataloader for everyone
# data loaders

batch_size_train = 32
batch_size_test = 32

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size_train, 
                                           shuffle=True)

val_loader = torch.utils.data.DataLoader(val_dataset,
                                           batch_size=batch_size_train, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=batch_size_test, 
                                          shuffle=True)

# import matplotlib.pyplot as plt
# # We can check the dataloader
# _, (example_datas, labels) = next(enumerate(train_loader))
# sample = example_datas[0]
# # show the data
# plt.imshow(sample.permute(1, 2, 0));
# plt.savefig('tf.png')


#Now it's time to create model
class newModelresnet34(nn.Module):
    def __init__(self, num_classes):
        super(newModelresnet34, self).__init__()
        
        self.model = models.resnet34(weights='IMAGENET1K_V1')
        
        nm_ft = self.model.fc.in_features
        # change the last layer of the classifier for a new dataset

        self.model.fc = nn.Linear(nm_ft, num_classes)

    def forward(self, x):
        
        x = self.model(x)

        return x

# define the model models 
model = newModelresnet34(40) # since STL-10 dataset has 10 classes, we set num_classes = 10
# device: cuda (gpu) or cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# map to device
model = model.to(device) # `model.cuda()` will also do the same job
# make the parameters trainable
for param in model.parameters():
    param.requires_grad = True

import torch.optim as optim
## some hyperparameters related to optimizer
learning_rate = 0.0001
weight_decay = 0.0005
# define optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

##define train function
def train(model, device, train_loader, optimizer):
    # meter
    loss = AverageMeter()
    # switch to train mode
    model.train()
    tk0 = tqdm(train_loader, total=int(len(train_loader)))
    for batch_idx, (data, target) in enumerate(tk0):
        # after fetching the data transfer the model to the 
        # required device, in this example the device is gpu
        # transfer to gpu can also be done by 
        # data, target = data.cuda(), target.cuda()
        data, target = data.to(device), target.to(device)  
        # compute the forward pass
        # it can also be achieved by model.forward(data)
        output = model(data) 
        # compute the loss function
        loss_this = F.cross_entropy(output, target)
        # initialize the optimizer
        optimizer.zero_grad()
        # compute the backward pass
        loss_this.backward()
        # update the parameters
        optimizer.step()
        # update the loss meter 
        loss.update(loss_this.item(), target.shape[0])
    print('Train: Average loss: {:.4f}\n'.format(loss.avg))
    return loss.avg
        
##define test function
def test(model, device, test_loader):
    # meters
    loss = AverageMeter()
    acc = AverageMeter()
    correct = 0
    # switch to test mode
    model.eval()
    for data, target in test_loader:
        # after fetching the data transfer the model to the 
        # required device, in this example the device is gpu
        # transfer to gpu can also be done by 
        # data, target = data.cuda(), target.cuda()
        data, target = data.to(device), target.to(device)  # data, target = data.cuda(), target.cuda()
        # since we dont need to backpropagate loss in testing,
        # we dont keep the gradient
        with torch.no_grad():
            # compute the forward pass
            # it can also be achieved by model.forward(data)
            output = model(data)
        # compute the loss function just for checking
        loss_this = F.cross_entropy(output, target) # sum up batch loss
        # get the index of the max log-probability
        pred = output.argmax(dim=1, keepdim=True) 
        # check which of the predictions are correct
        correct_this = pred.eq(target.view_as(pred)).sum().item()
        # accumulate the correct ones
        correct += correct_this
        # compute accuracy
        acc_this = correct_this/target.shape[0]*100.0
        # update the loss and accuracy meter 
        acc.update(acc_this, target.shape[0])
        loss.update(loss_this.item(), target.shape[0])
    print('Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        loss.avg, correct, len(test_loader.dataset), acc.avg))

num_epoch = 5
for epoch in range(1, num_epoch + 1):
    epoch_loss = train(model, device, train_loader, optimizer)
    # writer.add_scalar('training_loss', epoch_loss, global_step = epoch)
test(model, device, test_loader)

print("Done")



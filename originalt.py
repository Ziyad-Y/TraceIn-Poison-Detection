
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torchvision.models as models
from torchvision import datasets
import torch.nn.functional as F
from torch.utils.data import DataLoader

model=torch.load("resnet18_original")   
correct =0  
transform=transforms.ToTensor()
test=datasets.MNIST(root="data", train=False, download=True, transform=transform)
test_load = DataLoader(test,batch_size=256,shuffle=False)
correct = 0
total = 0
with torch.no_grad():  
    for images, labels in test_load:
        outputs = model(images)  
        _, predicted = torch.max(outputs, 1) 
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
accuracy = correct / total
print('Accuracy of the model on the test images: {:.2f}%'.format(100 * accuracy))
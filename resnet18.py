import torch  
import numpy as np   
import torchvision   
import torchvision.transforms as transforms  
import torch.optim as optim  
import torch.nn as nn   
import torchvision.models as models  
from torchvision import datasets    
from torch.utils.data import DataLoader
class resnet18_poisoned:
    def __init__(self, epochs=None):
        self.epochs=epochs     
       


    def train(self):
          
        pass
    
    def get_data(self, dataset):
        if dataset=="MNIST":
            train_set = datasets.MNIST(root='data',train=True,download=True,transform=transforms.ToTensor()) 
            train_loader = DataLoader(train_set , batch_size=64, shuffle=True)


        elif dataset=="CIFAR10":
            train_set =datasets.CIFAR10(root='data',train=True,download=True,transform=transforms.ToTensor())   
            train_loader = DataLoader(train_set , batch_size=64, shuffle=True)


        else:
            print("ERROR: Invalid data set")
            exit(-1)

        return train_loader,train_set   

    def label_poison(self):
        pass  

    def clean_label_poison(self):
        pass

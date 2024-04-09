import torch  
import numpy as np   
import torchvision   
import torchvision.transforms as transforms  
import torch.optim as optim  
import torch.nn as nn   
import torchvision.models as models  
from torchvision import datasets
class resnet18:
    def __init__(self, epochs=None):
        self.epochs=epochs     
       


    def train(self):
          
        pass
    
    def get_data(self, dataset):
        if dataset=="MNIST":
            train_set = datasets.MNIST(root='data',train=True,download=True,transform=transforms.ToTensor()) 


        elif dataset=="CIFAR10":
            train_set =datasets.CIFAR10(root='data',train=True,download=True,transform=transforms.ToTensor())   

        else:
            print("ERROR: Invalid data set")
            exit(-1)

        return train_set.targets

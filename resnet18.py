import torch  
import numpy as np   
import torchvision   
import torchvision.transforms as transforms  
import torch.optim as optim  
import torch.nn as nn   
import torchvision.models as models  
from torchvision import datasets    
from torch.utils.data import DataLoader
import random

class resnet18_poisoned:
    def __init__(self, epochs=None):
        self.epochs=epochs
        self.poisonIdx=[]  
        self.Original_dataset=None  
        self.Poisoned_dataset=None  
             
       


    def train(self):
          
        pass
    
    def get_data(self, dataset):
        if dataset=="MNIST":
            train_set = datasets.MNIST(root='data',train=True,download=True,transform=transforms.ToTensor()) 
            self.random_label_poison(train_set,0.1)
            train_loader = DataLoader(train_set , batch_size=64, shuffle=True)


        elif dataset=="CIFAR10":
            train_set =datasets.CIFAR10(root='data',train=True,download=True,transform=transforms.ToTensor())   
            train_loader = DataLoader(train_set , batch_size=64, shuffle=True)


        else:
            print("ERROR: Invalid data set")
            exit(-1)

        return train_loader,train_set   

    def random_label_poison(self, train_set,percentage, desired):
        random.seed(800)
        end = (percentage * len(train_set))
        count =0
        while True:
            i=random.randint(0, len(train_set)-1)
            if train_set.targets[i] != desired and i not in self.poisonIdx :
                train_set.targets[i] = desired   
                self.poisonIdx.append(i)
                count+=1    
            if count == end:
                break


    
    
    def target_label_poison(self, train_set, target, desired):
          for i in range(len(train_set)):
            if train_set.targets[i] == target:
                train_set.targets[i] = desired 
                self.poisonIdx.append(i)


    def clean_label_poison(self):
        pass


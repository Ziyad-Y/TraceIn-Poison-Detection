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



class Poison:
    
    def __init__(self, beta):      
        self.beta=beta  
        self.model=models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.model.conv1= nn.Conv2d(1,64,kernel_size=7,stride=1,padding=3,bias=False)  
        num_feat = model.fc.in_features  
        self.model.fc=nn.Linear(num_feat,10)   
        self.criterion=nn.MSELoss()

    def generate_poison(self,target_image,desired_image,lr):
        self.model.train()

        
        






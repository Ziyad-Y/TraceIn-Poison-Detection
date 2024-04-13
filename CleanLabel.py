import torch  
import numpy as np   
import torchvision   
import torchvision.transforms as transforms  
import torch.optim as optim  
import torch.nn as nn   
import torchvision.models as models  
from torchvision import datasets    
from torch.utils.data import DataLoader 

class Poison:
    
    def __init__(self, beta):
        self.model=models.resnet18(weights=models.ResNet18_Weights.DEFAULT )   
        self.optimizer = torch.optim.SGD(self.model.parameters(),lr=0.01)   
        self.Lp=nn.CrossEntropyLoss()
        self.beta=beta  

    def generate_poison(self,target_image,desired_image):
        self.model.train()  
        target_output = self.model(target_image)  
        desired_clone = desired_image.clone().detach().requires_grad_(True)  

        for _ in range(100):
            self.optimizer.zero_grad()  
            output=self.model(desired_clone)  
            loss=self.Lp(output,target_output.argmax().unsqueeze(0))  
            loss.backwards()   
            gradient= desired_clone.grad.data   
            desired_clone = (desired_clone - 0.1 * gradient) / (1 + self.beta * 0.1)  

        return desired_clone.detach()







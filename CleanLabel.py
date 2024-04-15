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
        self.beta = beta  
        self.model = torch.load("clean_resnet18")
        self.model.eval()
    
    def Loss(self, x, t):
        f_x = self.model(x)  
        f_t = self.model(t)   
        return torch.norm(f_x - f_t, p=2)

    def generate_poison(self, target_image, desired_image, lr):
        x = desired_image.clone().requires_grad_(True)
        optimizer = optim.SGD([x], lr=lr)
        
        for i in range(1000):
            optimizer.zero_grad()  
            loss = self.Loss(x, target_image)
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                x_hat = x - lr * x.grad if x.grad is not None else x
                x = (x_hat + lr * self.beta * desired_image) / (1 + self.beta * lr)    

        return x
            


       

        
        






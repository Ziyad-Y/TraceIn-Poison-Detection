import torch  
import numpy as np   
import torchvision   
import torchvision.transforms as transforms  
import torch.optim as optim  
import torch.nn as nn   
import torchvision.models as models  
from torchvision import datasets    
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt 
from CleanLabel import *
from PIL import Image
def grayscale_to_rgb(img):
    img_rgb = Image.merge('RGB', (img, img, img))
    return img_rgb

class train_poisoned:
    def __init__(self, epochs=None):
        self.epochs=epochs
        self.poisonIdx=[]
        self.trainset=None  
        self.testest=None   
        self.Train_Loader=None  
        self.Test_Loader=None   
        self.originalImages=[]
       


    def train(self):
          
        pass
    
    def get_data(self, dataset):
        weights= models.ResNet18_Weights.DEFAULT  
        preprocess = weights.transforms()
        transform=transforms.Compose([  
            transforms.Resize((224,224)),  
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()    
        ])
        
        if dataset=="MNIST":
            self.trainset = datasets.MNIST(root='data',train=True,download=True,transform=transform)  
            

        #implement posison if needed
        elif dataset=="CIFAR10":
            self.trainset =datasets.CIFAR10(root='data',train=True,download=True,transform=transforms)   
            
        else:
            print("ERROR: Invalid data set")
            exit(-1)

    def random_label_poison(self, percentage, desired):
        np.random.seed(800)
        remainder =[i for i in range(len(self.trainset)) if self.trainset.targets[i]!= desired]
        count =int(percentage * len(remainder))
        poison_indices = np.random.choice(len(remainder),count, replace=False)  
        for i in poison_indices:
            index = remainder[i]
            self.trainset.targets[index]=desired  
            self.poisonIdx.append(index)  

           
    
    def target_label_poison(self, target, desired):
          for i in range(len(self.trainset)):
            if self.trainset.targets[i] == target:
                self.trainset.targets[i] = desired 
                self.poisonIdx.append(i)  

          


    def clean_label_poison(self, target, beta):
        p = Poison(beta)
        temp = []
        desired_img = None
        for i in range(len(self.trainset)):
            if self.trainset.targets[i] == 9:
                desired_img,_ = self.trainset[i]
                break

        if desired_img is None:
            print("Desired image not found.")
            return
        
        for i in range(len(self.trainset)):
            if self.trainset.targets[i] == target:
                self.poisonIdx.append(i)
                image, label = self.trainset[i]
                poisoned_img = p.generate_poison(image, desired_img)
                temp.append((poisoned_img.squeeze(0), target))  
            else:
                temp.append(self.trainset[i])

        self.trainset = temp
                

            
       
       

        


              
              

        


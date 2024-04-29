import torch  
import numpy as np   
import os
import torchvision   
import torchvision.transforms as transforms  
import torch.optim as optim  
import torch.nn as nn   
import torchvision.models as models  
from torchvision import datasets    
import torch.nn.functional as F
from torch.utils.data import DataLoader 
from art.attacks.poisoning import FeatureCollisionAttack 
from art.estimators.classification import PyTorchClassifier  




class TraceIN_UTILS:
    def __init__(self, epochs=None):
        self.epochs=epochs
        self.poisonIdx=[]

       


   
 
        
        
    #random label flipping
    def random_label_poison(self, trainset,percentage):
        np.random.seed(800)
        count =int(percentage * len(trainset))
        poison_indices = np.random.choice(len(trainset),count, replace=False)  
        for i in poison_indices:
            image,label= trainset[i]       
            trainset[i] = (image,np.random.randint(0,10))  
            self.poisonIdx.append(i)
     
        

           
    #targeted label flipping
    def target_label_poison(self, trainset,target, desired):

        for idx,(image,label) in enumerate(trainset):
          if label == target:
            trainset[idx] = (image,desired) 
            self.poisonIdx.append(idx) 

        
        
        

          


    def clean_label_poison(self, trainset,target,base):
        model=models.resnet18()  
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3, bias=False)
        num_feat = model.fc.in_features
        model.fc = nn.Linear(num_feat, 10)
        state_dict=torch.load("Checkpoints/resnet18.pt") 
        model.load_state_dict(state_dict)
        model.to('cuda')   
       
        target_img = None  
        for image,label in trainset:
            if label == target:
                target_img = image.detach().clone()      
                break  
        target_img = target_img.cpu().detach().numpy()
        classifier = PyTorchClassifier(model=model, loss=nn.CrossEntropyLoss(), optimizer=optim.SGD(model.parameters(), lr=0.001), input_shape=[1,1,28,28],nb_classes=10, clip_values=(0,1))   
        attack=FeatureCollisionAttack(classifier,target_img.reshape(1,1,28,28),feature_layer='fc',max_iter=3000, similarity_coeff=15000, watermark=0.1, verbose=False) 
        
       
        for idx, (image,label) in enumerate(trainset):
            if label==base:
                self.poisonIdx.append(idx)   
                base_np = image.cpu().detach().numpy()  
                poison,poison_label = attack.poison(base_np.reshape(1,1,28,28))    
                tensor = torch.tensor(poison)   
                tensor=tensor.squeeze(0)   
                trainset[idx] = (tensor, label) 


    
    #train model
    def train(self):
        transform = transforms.ToTensor()
        trainset = datasets.MNIST(root='data', train=True, download=True, transform=transform)  
        train_loader = DataLoader(trainset, batch_size=256, shuffle=True)
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3, bias=False)
        num_feat = model.fc.in_features
        model.fc = nn.Linear(num_feat, 10)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

        print("Begin Train")
        epochs = 50
        for i in range(epochs):
            if (i + 1) % 5 == 0:
                print(f"Saving checkpoint {i+1}")
                checkpoint = {
                    'epoch': i + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }
                torch.save(checkpoint, f"Checkpoints/checkpoint{i}.pt")

            for batch_idx, (image, label) in enumerate(train_loader):
                output = model(image)
                loss = criterion(output, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        torch.save(model.state_dict(), "Checkpoints/resnet18.pt")  

    #train pertubed model
    def train_pert(self):
       transform = transforms.ToTensor()
       trainset = datasets.MNIST(root='data', train=True, download=True, transform=transform)  
       modified = [] 
       for image,label in trainset:
            modified.append((image,label))  
       self.clean_label_poison(modified,9,3)
       train_loader = DataLoader(modified, batch_size=256, shuffle=True)
       model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
       model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3, bias=False)
       num_feat = model.fc.in_features
       model.fc = nn.Linear(num_feat, 10)
       criterion = nn.CrossEntropyLoss()
       optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
       print("Begin Train")
       epochs = 50
       for i in range(epochs):
           if (i + 1) % 5 == 0:
               print(f"Saving checkpoint {i+1}")
               checkpoint = {
                   'epoch': i + 1,
                   'model_state_dict': model.state_dict(),
                   'optimizer_state_dict': optimizer.state_dict(),
                   'loss': loss,
               }
               torch.save(checkpoint, f"Perturbed/checkpoint{i}.pt")
           for batch_idx, (image, label) in enumerate(train_loader):
               output = model(image)
               loss = criterion(output, label)
               optimizer.zero_grad()
               loss.backward()
               optimizer.step()
       torch.save(model.state_dict(), "Perturbed/resnet18.pt")

 
    #test model

    def test(self):
        transform= transforms.ToTensor()
        testset=datasets.MNIST(root="data",train=False,download=True, transform=transform)  
        test_load=DataLoader(testset,batch_size=256,shuffle=False)  

        model=models.resnet18()  
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3, bias=False)
        num_feat = model.fc.in_features
        model.fc = nn.Linear(num_feat, 10) 

        state_dict=torch.load("Checkpoints/resnet18.pt") 
        model.load_state_dict(state_dict)

        correct =0  
        transform=transforms.ToTensor()
        correct = 0
        total = 0
        with torch.no_grad():  
            for images, labels in test_load:
                outputs = model(images)  
                _, predicted = torch.max(outputs, 1) 
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        print("Accuracy of the model on the test images: %0.2f"%(100 * accuracy))  

    def test_pet(self):
        transform= transforms.ToTensor()
        testset=datasets.MNIST(root="data",train=False,download=True, transform=transform)  
        test_load=DataLoader(testset,batch_size=256,shuffle=False)  

        model=models.resnet18()  
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3, bias=False)
        num_feat = model.fc.in_features
        model.fc = nn.Linear(num_feat, 10) 

        state_dict=torch.load("Perturbed/resnet18.pt") 
        model.load_state_dict(state_dict)

        correct =0  
        transform=transforms.ToTensor()
        correct = 0
        total = 0
        with torch.no_grad():  
            for images, labels in test_load:
                outputs = model(images)  
                _, predicted = torch.max(outputs, 1) 
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        print("Accuracy of the model on the test images: %0.2f"%(100 * accuracy))


        
                

            
       
       

        


              
              

        


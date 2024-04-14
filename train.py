from CleanLabel import *


class train_poisoned:
    def __init__(self, epochs=None):
        self.epochs=epochs
        self.poisonIdx=[]
        self.trainset=None  
        self.testset=None   
        self.Train_Loader=None  
        self.Test_Loader=None   
        self.originalImages=[]
       


   
    
    def get_data(self):
        trasnform=transforms.ToTensor()
        self.trainset = datasets.MNIST(root='data',train=True,download=True,transform=transform)
        self.testset = datasets.MNIST(root='data',train=True,download=True,transform=transform)  
        
        

    def random_label_poison(self, percentage, desired):
        np.random.seed(800)
        remainder =[i for i in range(len(self.trainset)) if self.trainset.targets[i]!= desired]
        count =int(percentage * len(remainder))
        poison_indices = np.random.choice(len(remainder),count, replace=False)  
        for i in poison_indices:
            index = remainder[i]
            self.trainset.targets[index]=desired  
            self.poisonIdx.append(index)
        self.Train_Loader=DataLoader(self.trainset,batch_size=256,shuffle=True) 
        self.Test_Loader=DataLoader(self.testset)  

           
    
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
                poisoned_img = p.generate_poison(image, desired_img,0.1)
                temp.append((poisoned_img.squeeze(0), target))  
            else:
                temp.append(self.trainset[i])

        self.trainset = temp
                

            
       
       

        


              
              

        


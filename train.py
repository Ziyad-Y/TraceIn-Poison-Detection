from CleanLabel import *
import os

class train_poisoned:
    def __init__(self, epochs=None):
        self.epochs=epochs
        self.poisonIdx=[]

       


   
 
        
        

    def random_label_poison(self, trainset,percentage, desired):
        np.random.seed(800)
        remainder =[i for i, (image,label) in enumerate(trainset) if label!= desired]
        count =int(percentage * len(remainder))
        poison_indices = np.random.choice(len(remainder),count, replace=False)  
        for i in poison_indices:
            index = remainder[i]
            image,label= trainset[index]       
            trainset[index] = (image,desired)  
            self.poisonIdx.append(index)
     
        

           
    
    def target_label_poison(self, trainset,target, desired):

        for idx,(image,label) in enumerate(trainset):
          if label == target:
            trainset[idx] = (image,desired) 
            self.poisonIdx.append(idx) 

        
        
        

          


    def clean_label_poison(self, trainset,target, beta):
        p = Poison(beta)
        desired_img = None
        for image,label in trainset:
            if label == 9:
                desired_img= image.clone()
                break

        if desired_img is None:
            print("Desired image not found.")
            return
        
        for i, (image,label) in enumerate(trainset):
            if label == target:
                #print(i)
                self.poisonIdx.append(i)

                poisoned_img = p.generate_poison(image.view(1,1,28,28), desired_img.view(1,1,28,28),0.01)
                trainset[i]=(poisoned_img.squeeze(dim=0), label) 
          

     
        

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


        
                

            
       
       

        


              
              

        


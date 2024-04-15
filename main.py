from train import *
import matplotlib.pyplot as plt
import glob
import csv
r=train_poisoned()

transform = transforms.ToTensor() 

train= datasets.MNIST(root='data',train=True,download=True,transform=transform)    
test= datasets.MNIST(root='data',train=False,download=True,transform=transform)   

x=np.random.choice(len(train),100,replace=False)   
y=np.random.choice(len(test),20, replace=False)

train_points=[]  
test_points=[]   


for i in range(len(x)):
    image,label = train[x[i]]    
    train_points.append((image,label))   


for i in range(len(y)):
    image,label = test[y[i]]   
    test_points.append((image,label))   

r.clean_label_poison(train_points,3,0.25)
f = open("clean","w")   
w= csv.writer(f)   
w.writerow(["Train","Test", "TraceINCP"])

Checkpoints = glob.glob("Checkpoints/checkpoint*")     
model=models.resnet18()  
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3, bias=False)
num_feat = model.fc.in_features
model.fc = nn.Linear(num_feat, 10) 
optimizer = torch.optim.SGD(model.parameters(),lr=0.001)
TraceCP_L = []
for i,(z,z_l) in enumerate(train_points):
    for j,(zp, zp_l) in enumerate(test_points):
        TraceINCP=0
        for c in Checkpoints:
            state_dict=torch.load(c)
            
            model_state=state_dict['model_state_dict']   
            model.load_state_dict(model_state)
            model.eval()
            optimizer.load_state_dict(state_dict['optimizer_state_dict'])   
            loss = nn.CrossEntropyLoss()  

            eta =optimizer.param_groups[0]["lr"]  
            

            optimizer.zero_grad()   

            out_1 =model(z.unsqueeze(0))
            loss_1=loss(out_1,torch.tensor([z_l]))
            loss_1.backward()
            z_grad=[param.grad.clone() for param in model.parameters()]
            
            out_2 = model(zp.unsqueeze(0)) 
            loss_2=loss(out_2,torch.tensor([zp_l]))  
            loss_2.backward()
            zp_grad =[param.grad.clone() for param in model.parameters() ]

            product = sum(torch.sum(zg*zpg) for zg,zpg in zip(z_grad,zp_grad))   
            TraceINCP+= product*eta  
        w.writerow([i,j,TraceINCP.item()])   
        if TraceINCP.item() > 300:
            TraceCP_L.append((i,TraceINCP.item()))

f.close()   
unique=[]
for idx,val in TraceCP_L:
    if idx not in unique:
        unique.append(idx)   
success=0
for i in unique:
    if i in r.poisonIdx:
        success+=1   

print(success/len(r.poisonIdx))   
    
print(r.poisonIdx)
            





           








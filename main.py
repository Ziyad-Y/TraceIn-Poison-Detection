from train import *
import matplotlib.pyplot as plt
r = train_poisoned()  
r.get_data("MNIST")
#r.clean_label_poison(3,0.5)
#
#batch_size = labels_loader.batch_size
#total_samples = len(labels_loader.dataset)
#total_batches = np.ceil(total_samples / batch_size)
#
#
#
## Print the first 15 labels   
#new_label=[i *2 for i in range(15)]
#
#for i in range(15):
#    dataset.targets[i]=new_label[i]

model=models.resnet18(weights=models.ResNet18_Weights.DEFAULT) 
model.conv1= nn.Conv2d(1,64,kernel_size=7,stride=1,padding=3,bias=False)  
num_feat = model.fc.in_features  
model.fc=nn.Linear(num_feat,10)
loss=nn.MSELoss()  

model.train()
image, label = r.trainset[0]
image2,label = r.trainset[55]

x=image.view(1,1,28,28)  
x2=image.view(1,1,28,28)
l=loss(x,x2)  
print(l)

# Convert the PyTorch tensor to a NumPy array and then to a PIL image
image_np = image.numpy()
image_np = image_np.transpose((1, 2, 0))  # Convert from CxHxW to HxWxC
plt.imshow(image_np, cmap='gray')  
plt.title(f"Label: {label}")
plt.axis('off')  # Hide axes
plt.show()  

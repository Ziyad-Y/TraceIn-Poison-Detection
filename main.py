   
from resnet18 import *  
import matplotlib.pyplot as plt
r = resnet18_poisoned()
labels_loader, dataset = r.get_data("MNIST")

batch_size = labels_loader.batch_size
total_samples = len(labels_loader.dataset)
total_batches = np.ceil(total_samples / batch_size)

print("Total Samples:", total_samples)
print("Batch Size:", batch_size)
print("Total Batches:", total_batches)

# Print the first 15 labels   
new_label=[i *2 for i in range(15)]

for i in range(15):
    dataset.targets[i]=new_label[i]

for i in range(15):
    print(dataset.targets[i].item())   


print(len(dataset))   


image, label = dataset[13]

# Convert the PyTorch tensor to a NumPy array and then to a PIL image
image_np = image.numpy()
image_np = image_np.transpose((1, 2, 0))  # Convert from CxHxW to HxWxC
plt.imshow(image_np, cmap='gray')  
plt.title(f"Label: {label}")
plt.axis('off')  # Hide axes
plt.show()
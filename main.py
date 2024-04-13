from resnet18 import *
r = train_poisoned()  
r.get_data("MNIST")
r.clean_label_poison(3,0.5)
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


image, label = r.trainset[0]
print(image.shape)

# Convert the PyTorch tensor to a NumPy array and then to a PIL image
image_np = image.numpy()
image_np = image_np.transpose((1, 2, 0))  # Convert from CxHxW to HxWxC
plt.imshow(image_np, cmap='gray')  
plt.title(f"Label: {label}")
plt.axis('off')  # Hide axes
plt.show()  

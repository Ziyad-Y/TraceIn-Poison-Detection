   
from resnet18 import *  

r=resnet18()
labels=r.get_data("MNIST")      

print(labels[:10])
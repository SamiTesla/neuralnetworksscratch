from nnfs.datasets import spiral_data
import  numpy as np
import nnfs 
import full_layer_dense
import matplotlib.pyplot as plt
inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]
output = np.maximum(0, inputs)


#create class for the activation function
class Activation_ReLU():
    #forwardpass
    def forward(self,input):
        #calculate values from input
        self.output=np.maximum(0,input)
#create dataset
X,y= spiral_data(samples=100, classes=3)

#create a dense layer which accepts 2 inputs has 3 outputs

dense1=full_layer_dense.layer_dense(2,3)

#ReLU activation func
activation1=Activation_ReLU()
#forward pass the training data
dense1.forward(X)
#forward pass again and then also take the training data output from dense1
activation1.forward(dense1)

print(activation1[:5])
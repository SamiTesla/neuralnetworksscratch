from nnfs.datasets import spiral_data
import  numpy as np
import nnfs 
import matplotlib.pyplot as plt
#sets random seed to zero and allows us to ensure repeatability for our dataset going forth
nnfs.init()

class layer_dense():
    #we do it in (input,n_neuron) to avoid having to transpose later on(it saves time and effort by setting it up to already be compatible)
    def __init__(self,n_input,n_neuron):
        #init weights and biases
        #generating random weights
        #.random.randn gives us a gaussian distribution
        self.weights=0.01*np.random.randn(n_input,n_neuron)
        #sets biases to zero
        
        self.biases=np.zeros((1,n_neuron))
    def forward(self,input):
        #calculuate output from inputs, weights, and biases :)
        self.output=np.dot(input,self.weights)+self.biases
#set up our data
X,y= spiral_data(samples=100, classes=3)
#creates a dense layer which takes 2 inputs and outputs 3 values
dense1=layer_dense(2,1)
#forward pass of our data
dense1.forward(X)

print(dense1.output[:5])


    
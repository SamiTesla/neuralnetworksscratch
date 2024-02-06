import matplotlib.pyplot as plt
import nnfs
import numpy as np
from nnfs.datasets import vertical_data 
import spiral3
nnfs.init()

X,y = vertical_data(samples=100, classes=3)

#make model
dense1=spiral3.layer_dense(2,3)# 2 input 3 output
activation1= spiral3.Activation_ReLU()
dense2= spiral3.layer_dense(3,3) # 3input 3 output 
activation2=spiral3.Activation_Softmax()

#loss function
loss_func=spiral3.Loss_CategoricalCrossentropy()

#helper var
lowest_loss=9999999 #some init val
best_dense1_weight=dense1.weights.copy()
best_dense1_biases=dense1.biases.copy()
best_dense2_weight=dense2.weights.copy()
best_dense2_biases=dense2.biases.copy()

for iteration in range(10000):
    #new set of weights to iterate
    dense1.weight= 0.05*np.random.randn(2,3)
    dense1.biases=0.05*np.random.randn(1,3)
    dense2.weight=0.05*np.random.randn(3,3)
    dense2.biases=0.05* np.random.randn(1,3)
    #forward pass the training data thru layer
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.forward)

    #perform a forward pass thru activation functin
    #take the output of seccond dense layer here and return the loss
    loss= loss_func.calculate(activation2.output,y)

    #calculate accuracy from output of activation2 and its targets
    #calculate val along first axis
    predictions=np.argmax(activation2.output,axis=1)
    accuracy=np.mean(predictions==y)

    #should the loss be smaller print & save weights and biases 
if loss< lowest_loss:
    print('new set of weights found, iteration', iteration,'loss:', 'accuracy:', accuracy )
    best_dense1_weight=dense1.weight.copy()
    best_dense1_biases=dense1.biases.copy()
    best_dense2_weight=dense2.weight.copy()
    best_dense1_weight=dense2.biases.copy()
    lowest_loss= loss

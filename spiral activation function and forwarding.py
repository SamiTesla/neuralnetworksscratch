import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

#dense layer 
class layer_dense:
    #layer init
    def __init__(self,n_input,n_neuron):
        #init weights and biases
        self.weights=0.01 * np.random.randn(n_input,n_neuron)
        self.biases=np.zeros((1,n_neuron))
    #forward pass
    def forward(self,input):
        #calc output values from inputs, weight, and biases
        self.output=np.dot(input,self.weights)+ self.biases
#ReLu activation
class Activation_ReLU:
    #forward pass
    def forward(self,input):
        #calc output val from input
        self.output=np.maximum(0,input)
#softmax activation
class Activation_Softmax:
    #forward pass
    def forward(self, input):
        #unnormalized probabilities 
        exp_val=np.exp(input-np.max(input, axis=1, keepdims=True))
        #normalize for each sample 
        probabilities=exp_val/np.sum(exp_val, axis=1, keepdims=True)

        self.output=probabilities
#basic loss class
class loss:
    #calculate data and regularlization losses
    #given model output and ground truth vals
    def calculate(self, output, y):
        #calculate samples
        sample_losses=self.forward(output,y)
        #mean loss calc
        data_loss= np.mean(sample_losses)
        #return loss
        return data_loss
    #add more code in later chapters 

# cross entropy loss
class Loss_CategoricalCrossentropy(loss):
    #forward pass
    def forward(self, y_pred, y_true):
        # of samples in batch
        samples=len(y_pred)

        #clip data to prevent division by 0
        #clip both sides to not drag mean toward any val
        y_pred_clipped=np.clip(y_pred,1e-7, 1-1e-7)

        #prob for target val
        #only if categorical lables
        if len(y_true.shape)==1:
            correct_confidence=y_pred_clipped[
                range(samples), y_true]

        #mask values for one-hot encoded labels 
        elif len(y_true.shape)==2:
            correct_confidence=np.sum(y_pred_clipped*y_true, axis=1)
        #losses
        negative_log_likelihood= -np.log(correct_confidence)
        return negative_log_likelihood

X,y=spiral_data(samples=100, classes=3)

#dense layer w/ 2 inputs 3 outputs
dense1=layer_dense(2, 3)
#ReLU activation(use w/dense layer)
activation1=Activation_ReLU()
#create another dense layer w/ 3 inputs as the output from previous layer and 3 outputs
dense2=layer_dense(3, 3)

#softmax activation use w/ dense layer
activation2=Activation_Softmax()

#loss function creation
loss_func=Loss_CategoricalCrossentropy()

#forward pass the training data through the layer 
dense1.forward(X)

#make a forward pass through activation func, take output of first dense layer
activation1.forward(dense1.output)

#forward pass the second dense layer
#take output of activation func of first layer as input

dense2.forward(activation1.output)

#activation func forward pass, take second dense layer output
activation2.forward(dense2.output)

#output samples
print(activation2.output[:5])

#forward pass thru loss function
#take output of second dense layer here and return loss
loss=loss_func.calculate(activation2.output,y)

#print loss val
print('loss',loss)

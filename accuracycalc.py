import numpy as np 

#probability of 3 samples

softmax_output= np.array([[0.7, 0.2, 0.1], [0.5, 0.1, 0.4], [0.02, 0.9, 0.08]])

#target (ground truth) labels for 3 samples 
class_targets=np.array([0,1,1])

#calculate values on seccond axis(axis index 1)
predict= np.argmax(softmax_output,axis=1)

#check if target is hot encoded and convert
if len(class_targets.shape)==2:
    class_targets=np.argmax(class_targets, axis=1)

#true evals to 1; false to 0
accuracy = np.mean(predict==class_targets)

print('acc', accuracy)

#calculate accuracy from output of activation 2 and target
#calc vals along first axis
predict=np.argmax(activation2.output, axis=1)
if len(y.shape)==2:
    y=np.argmax(y, axis=1)
accuracy=np.mean(predict==y)
print('acc', accuracy)

import numpy as np
#this is meant to check if a vector is hot encoded
#lists are sparse, but a list of list(lol) will have one set of hot encoded vectors
#zero everything except whats at the correct labels 

softmax_output=np.array([[0.7,0.1,0.2],[0.1,0.5,0.4],[0.02,0.9,0.08]])

class_targets =np.array([[1,0,0],[0,1,0],[0,1,0]])

#probability for target vals
#only if catagory labels

if len(class_targets.shape)==1:
    correct_confidence=softmax_output[
        range(len(softmax_output)), class_targets
    ]
#masks values only for one hot encoded labels
elif len(class_targets.shape)==2:
    correct_confidence=np.sum(softmax_output*class_targets, axis=1)
#losses
neg_log=-np.log(correct_confidence)

average_loss=np.mean(neg_log)

print(average_loss)

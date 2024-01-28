import numpy as np

# we are seeing an networks confidence using target values to represent a dog,cat,cat

softmax_output=[[0.7, 0.1, 0.2],
    [0.1, 0.5, 0.4],
    [0.02, 0.9, 0.08]]

class_targets=[0,1,1] #dog cat cat

for targ_indx, distribution in zip(class_targets, softmax_output):
    #print confidence score 
    print(distribution[targ_indx])

#with numpy
    
softmax_output=np.array([[0.7, 0.1, 0.2],
    [0.1, 0.5, 0.4],
    [0.02, 0.9, 0.08]])

#confidence list
print(softmax_output[[0,1,2], class_targets])

#apply negative log 
neg_log=-np.log(softmax_output[range(len(softmax_output)), class_targets])

average_loss=np.mean(neg_log)
print(average_loss)

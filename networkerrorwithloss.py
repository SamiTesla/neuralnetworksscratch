import math
#cross entropy calculation
#example output from neural network
#purpose: model’s confidence about these predictions is high only for one of them. 
#The Categorical Cross-Entropy Loss accounts for that and outputs a larger loss the lower the confidence is(we only care about the confidence)
softmax_output=[0.7,0.1,0.2]

#ground truth
target_output=[1,0,0]

#calculate
loss= -math.log((softmax_output[0]*target_output[0]
                 +softmax_output[1]*target_output[1]
                 + softmax_output[2]*target_output[2] ))
print(loss)

#notice how [1,2] are multiplied by 0 in this situation we can instead do
losss=-math.log(softmax_output[0])
print(loss)

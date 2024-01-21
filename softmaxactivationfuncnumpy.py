import numpy as np

layer_outputs=[4.8, 1.21, 2.385]

#for each value in the vector calculate the exponential value 
exp_value=np.exp(layer_outputs)

print('exponentiated values')
print(exp_value)

#normalize values
norm_values=exp_value/sum(exp_value)
print('normalized exponent values')
print(norm_values)
print('sum of normal values', sum(norm_values))

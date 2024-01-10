inputs = [1, 2, 3, 2.5]
weights = [[0.2, 0.8, -0.5, 1],
[0.5, -0.91, 0.26, -0.5],
[-0.26, -0.27, 0.17, 0.87]]
biases = [2, 3, 0.5]
#output of the currentlayer 
layer_outputs=[]
#individual neurons
for neuron_weights, neuron_bias in zip(weights,biases):
    #zero output of a given neuron
    neuron_output = 0
    #each input and weight added to the neuron
    for n_input, weight in zip(inputs,neuron_weights):
    #multiply input with its associated weight
    #add to the output variable
        neuron_output+=n_input*weight
    #add the bias
    neuron_output+= neuron_bias
    #add the output
    layer_outputs.append(neuron_output)
print(layer_outputs)

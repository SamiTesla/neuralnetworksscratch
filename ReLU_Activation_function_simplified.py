input=[0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]

output=[]
#this is for the function
for i in input:
    #max allows us to simplify the function, by setting our range of values to be 0 or i(has to be greater than 0) 
    output.append(max(0,i))
print(output)
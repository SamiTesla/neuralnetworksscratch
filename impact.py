import matplotlib.pyplot as plt
import numpy as np
def f(x):
    return 2*x**2
x=np.array(range(5));
y=f(x)


print(y[2]-y[1]/x[2]-x[1])
print((y[1]-y[0])/(x[1]-x[0]))
def f(x):
    return 2*x**2
y=f(x)
print(x)
print(y)

p2delta=.0001
x1=1
x2=x1+p2delta
y1=f(x1) #result at deriviative 
y2=f(x2)# result at other, point 

approxderive=(y2-y1)/(x2-x1)
print(approxderive)
plt.plot(x,y)
plt.show()

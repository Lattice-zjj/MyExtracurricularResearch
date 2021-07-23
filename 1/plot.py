import numpy as np
import matplotlib.pyplot as plt

x=np.array([0,0,0,0.25,0.25,0.5,0.75,0.75,1])
y=np.array([0,0.25,0.5,0.5,0.75,0.75,0.75,1,1])
x2=np.array([0,0,0.25,0.25,0.5,0.75,0.75,0.75,1])
y2=np.array([0,0.25,0.25,0.5,0.5,0.5,0.75,1,1])
#plt.plot(x,y,label='C1')
plt.plot(y2,x2,label='C2')
plt.title('ROC curves for classifier C2')
plt.legend()
plt.show()



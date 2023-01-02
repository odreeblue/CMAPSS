
import matplotlib.pyplot as plt

import numpy as np
from matplotlib import pyplot as plt





plt.title('1')
x1= np.array([1,3,6,4])
y1 = np.array([10,20,30,40])
plt.plot(x1,y1)


x2= np.array([1,3,6,4])
y2 = np.array([1,3,6,4])
plt.plot(x2,y2)
plt.savefig('./temp/test1.png')


plt.clf()

plt.title('2')
x1= np.array([1,6,0,4])
y1 = np.array([10,20,30,40])
plt.plot(x1,y1)
plt.grid()

x2= np.array([1,3,6,4])
y2 = np.array([1,3,6,4])
plt.plot(x2,y2)

plt.show()

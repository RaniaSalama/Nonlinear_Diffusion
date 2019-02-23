import numpy as np
import matplotlib.pyplot as plt
from pylab import *
 
p1 = np.loadtxt("p=1.0_variances_100.txt")
p5 = np.loadtxt("p=0.5_variances_100.txt")


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

cnt = 0
for i in range(10):
   for j in range(10):
        if p5[i, j] > 1000:
           continue
        if cnt == 0:
            plt.scatter(i, np.log(p5[i, j]), color='r', label="power")
            cnt = 1
        else:
            plt.scatter(i, np.log(p5[i, j]), color='r')

cnt = 0
for i in range(10):
   for j in range(10):
         if cnt == 0:
            plt.scatter(i, np.log(p1[i, j]), color='b', label="hk")
            cnt = 1
         else:
            plt.scatter(i, np.log(p1[i, j]), color='b')

plt.xticks(np.arange(10), ('T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot'))
plt.xticks(rotation=45)
plt.ylabel("log(Standard Deviation)")
plt.xlabel("Classes")
plt.legend(loc='upper right')

#major_ticks_y = np.arange(0, 610, 200)
#major_ticks_x = np.arange(0, 0.5, 10)

#ax.set_xticks(major_ticks_x)
#ax.set_yticks(major_ticks_y)
#ax.grid(which='major', alpha=0.2, linestyle='dashed')


plt.show()

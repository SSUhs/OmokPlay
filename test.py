import re
import numpy as np


a = np.zeros([15,15])
b = np.zeros([15,15])
c = [a,b]
a.transpose()
print(c)
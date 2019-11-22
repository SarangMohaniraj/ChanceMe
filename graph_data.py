import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from preprocessing import plot,college

# plot data
X = np.array(plot[["ACT","GPA"]])
y = np.array(plot["Result"])
admitted = X[np.argwhere(y==1)]
rejected = X[np.argwhere(y==0)]
plt.scatter([p[0][0] for p in rejected], [p[0][1] for p in rejected], s = 25, marker="x", color = 'red')
plt.scatter([p[0][0] for p in admitted], [p[0][1] for p in admitted], s = 25, marker="x", color = 'green')
plt.title(college + " Scattergram")
plt.xlabel("ACT")
plt.ylabel("GPA")
plt.gca().set(xlim=(0, 37), ylim=(0, 5))
plt.show()
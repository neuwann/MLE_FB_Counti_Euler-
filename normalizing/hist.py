import matplotlib.pyplot as plt
import numpy as np
import csv
import pprint
from mpl_toolkits.mplot3d import Axes3D
plt.style.use('ggplot')
plt.rcParams["axes.facecolor"] = 'white'
# Fixing random state for reproducibility
np.random.seed(19990811)

with open("data.csv") as f:
    print(f.reas())


#x, y = np.random.rand(2, 100) * 4
#hist, xedges, yedges = np.histogram2d(x, y, bins=4, range=[[0, 4], [0, 4]])

# Construct arrays for the anchor positions of the 16 bars.
#xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
#xpos = xpos.ravel()
#ypos = ypos.ravel()
#zpos = 0

# Construct arrays with the dimensions for the 16 bars.
#dx = dy = 0.5
#dz = hist.ravel()

#fig = plt.figure(figsize=(5,5))
#ax = fig.add_subplot(111, projection='3d')
#ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')
#plt.savefig("3d_histogram.jpg",dpi=120)
#plt.show()
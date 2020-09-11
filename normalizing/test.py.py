import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
plt.style.use('ggplot')
plt.rcParams["axes.facecolor"] = 'white'
# Fixing random state for reproducibility
np.random.seed(19990811)


x, y = np.random.rand(2, 100) * 4
hist, xedges, yedges = np.histogram2d(x, y, bins=4, range=[[0, 4], [0, 4]])

# Construct arrays for the anchor positions of the 16 bars.
xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
xpos = xpos.ravel()
ypos = ypos.ravel()
zpos = 0

# Construct arrays with the dimensions for the 16 bars.
dx = dy = 0.5
dz = hist.ravel()

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111, projection='3d')
ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')
plt.savefig("3d_histogram.jpg",dpi=120)
plt.show()

fig = plt.figure(figsize=(10,3))
dx=dy=0.25
ax1 = fig.add_subplot(131, projection='3d')
ax1.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')
ax1.set_title('dx=dy=0.25')
dx=dy=0.5
ax2 = fig.add_subplot(132, projection='3d')
ax2.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')
ax2.set_title('dx=dy=0.5')

dx=dy=0.75
ax3 = fig.add_subplot(133, projection='3d')
ax3.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')
ax3.set_title('dx=dy=0.75')

plt.savefig("3d_histogram_dxdy.jpg",dpi=120)
plt.show()

#bins

hist, xedges, yedges = np.histogram2d(x, y, bins=4, range=[[0, 4], [0, 4]])
xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
xpos = xpos.ravel()
ypos = ypos.ravel()
dx = dy = 0.5
dz = hist.ravel()

hist8, xedges8, yedges8 = np.histogram2d(x, y, bins=8, range=[[0, 4], [0, 4]])
xpos8, ypos8 = np.meshgrid(xedges8[:-1] + 0.125, yedges8[:-1] + 0.125, indexing="ij")
xpos8 = xpos8.ravel()
ypos8 = ypos8.ravel()
dx8 = dy8 = 0.25
dz8 = hist8.ravel()

fig = plt.figure(figsize=(9,4))
ax1 = fig.add_subplot(121, projection='3d')
ax1.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')
ax1.set_title('bins=4')
ax2 = fig.add_subplot(122, projection='3d')
ax2.bar3d(xpos8, ypos8, zpos, dx8, dy8, dz8, zsort='average')
ax2.set_title('bins=8')

plt.savefig("3d_histogram_bins.jpg",dpi=120)
plt.show()

fig = plt.figure(figsize=(10,3))
ax1 = fig.add_subplot(131, projection='3d')
ax1.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')
ax1.set_title('zsort=average')
ax2 = fig.add_subplot(132, projection='3d')
ax2.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='min')
ax2.set_title('zsort=min')
ax3 = fig.add_subplot(133, projection='3d')
ax3.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='max')
ax3.set_title('zsort=max')

plt.savefig("3d_histogram_zsort.jpg",dpi=120)
plt.show()

 

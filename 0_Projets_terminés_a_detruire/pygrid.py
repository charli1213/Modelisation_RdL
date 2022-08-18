import numpy as np
import matplotlib.pyplot as plt
import pygridgen

# ====== Functions calls ========
def create_maps(xpos,ypos) :
    "gets you a vector with the position of each nodes (maps)"
    nx = xpos.shape[1]
    ny = xpos.shape[0]
    maps = np.zeros(xpos.shape)
    inode = 0
    for j in range(ny) :
        for i in range(nx) :
            maps[j,i] = int(inode)
            inode += 1
    return maps

# -------------------------------


# -------------------------------
def get_mapping(maps, xpos, ypos) :
    """Get you a dict with the position of each nodes (mapping)"""
    return {i:(j,k) for i,j,k in zip(maps.flatten(),
                                     grid.x.flatten(),
                                     grid.y.flatten())}
# -------------------------------


# -------------------------------
def extract_shapes(maps) :
    """ Extract shapes from maps (vector), since curvilinear."""
    nx = maps.shape[1]
    ny = maps.shape[0]
    shapes = {}
    ishape = 0
    for j in range(ny-1) :
        for i in range(nx-1) :
            shapes[ishape] = [maps[j,i],
                             maps[j,i+1],
                             maps[j+1,i+1],
                             maps[j+1,i]]
            ishape += 1
    return shapes
# -------------------------------



# ================================ DATA


fig, data_ax = plt.subplots(figsize=(8.5, 5), ncols=1)
#plt.grid(linestyle='dashed')




x = np.array([0.5, 2.0, 3.5, 2.5, 1.5, 0.5])
y = np.array([0.5, 0.5, 1.0, 2.0, 2.0, 0.5])
beta = np.array([1, 1, 0, 1, 1, 0])
grid = pygridgen.Gridgen(x, y, beta, shape=(10, 10))



# ================================ Test ugrid
maps = create_maps(grid.x,grid.y)
mapping = get_mapping(maps, grid.x, grid.y)
shapes = extract_shapes(maps)

for values in shapes.values() :
    # Getting positions of each nodes for each shapes
    xquad = [mapping[inode][0] for inode in values]
    yquad = [mapping[inode][1] for inode in values]
    # Plotting shapes-frontiers : 
    data_ax.fill(xquad,yquad,
                 color = 'black',
                 linestyle = 'dotted',
                 linewidth = 0.3,
                 facecolor='aliceblue')

    
# Plotting each node numbers :
for ix, iy, inode in zip(grid.x_rho.flatten(),
                         grid.y_rho.flatten(),
                         list(shapes.keys())) : 
    data_ax.annotate(int(inode), (ix,iy), ha='center', va='center', color='blue')



# Plotting grid frontier and nodes. 
data_ax.plot(x,y, c='black', label='Contours', linewidth = 1)
#data_ax.scatter(grid.x.flatten(),grid.y.flatten(), c='black', label = 'Nodes')
#data_ax.scatter(grid.x_rho,grid.y_rho, c='blue', marker = 'D', label = 'Centroids') # Pas nécessaire.


# Figure fine tunning
data_ax.set_axisbelow(True)
data_ax.legend()
fig.tight_layout()
fig.show()

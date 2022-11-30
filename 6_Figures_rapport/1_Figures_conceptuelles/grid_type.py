# ==== IMPORTATION DES MODULES ====
import numpy as np
import matplotlib.pyplot as plt
import pygridgen
from scipy import interpolate
from scipy.interpolate import CubicSpline

# ==== A. CREATION DES FONCTIONS ====
# ------------------------------
def create_maps(xpos,ypos) :
    """Fonction qui utilise les positions (x,y) de chaque noeuds
    pour créer un vecteur (carte/maps) qui associe chaque noeuds 
    à un nombre (INT). Le vecteur résultant possède la même forme
    que les grilles entrantes."""
    
    nx = xpos.shape[1]
    ny = xpos.shape[0]
    maps = np.zeros(xpos.shape)
    inode = 0
    for j in range(ny) :
        for i in range(nx) :
            maps[j,i] = int(inode)
            inode += 1
    return maps
# ------------------------------


# ------------------------------
def get_mapping(maps, xpos, ypos) :
    """Fonction qui prend la carte (np.array) et les positions
    (x,y) de chaque noeuds et retourne un dictionnaire avec
    comme clef (keys) l'identifiant du noeuds et comme valeur
    (value) un tuple de position (x,y)"""
    
    return {i:(j,k) for i,j,k in zip(maps.flatten(),
                                     grid.x.flatten(),
                                     grid.y.flatten())}
# ------------------------------


# ------------------------------
def extract_shapes(maps) :
    """Fonction qui crée les quadrilatères à l'aide de la carte
    précédente et retourne un np.array contenant toutes les
    connections entre chaque noeuds (shapes)."""
    
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




# ==== B. CRÉATION D'UNE FIGURE ====
fig, axes = plt.subplots(figsize=(9, 3.5), ncols=3)




# ==== C. CREATION GRILLE RECTILINÉAIRE STRUCTURÉE ====
x0    = np.array([0,5,5,0,0])
y0    = np.array([5,5,0,0,5])
beta0 = np.array([1,1,1,1,0])
shape0 = (6,6)


# Création de la grille + extraction des données de grille : 
grid = pygridgen.Gridgen(x0, y0, beta0, shape=shape0)
maps = create_maps(grid.x,grid.y)
mapping = get_mapping(maps, grid.x, grid.y)
shapes = extract_shapes(maps)

# Plotting each shapes :
for values in shapes.values() :
    # Getting positions of each nodes for each shapes
    xquad = [mapping[inode][0] for inode in values]
    yquad = [mapping[inode][1] for inode in values]
    # Plotting shapes-frontiers : 
    axes[0].fill(xquad,yquad,
                 color = 'black',
                 linestyle = 'dotted',
                 linewidth = 0.3,
                 facecolor='aliceblue')

# Plotting nodes
axes[0].scatter(grid.x,grid.y,color='black',s=8)
    
# Plotting each node numbers :
for ix, iy, inode in zip(grid.x_rho.flatten(),
                         grid.y_rho.flatten(),
                         list(shapes.keys())) :
    axes[0].annotate(int(inode),(ix,iy),
                     ha='center',va='center',c='blue',fontsize=6)

# Plotting contours :
axes[0].plot(x0,y0, c='black', label='Contours', linewidth = 1)

# Figure fine-tunning: 
axes[0].set_axisbelow(True)
axes[0].set_title("Grille rectilinéaire\nstructurée")




# ==== D. CRÉATION GRILLE NON-STRUCTURÉE ====

xtop = np.array([0,1,2.5,4,5])
xbot = np.array([5,4,2.5,1,0])
ytop = np.array([4,4.7,5,4.7,4])
ybot = np.array([0.5,0,0.5,1,1])
interptop = interpolate.interp1d(xtop,ytop,kind='cubic')
interpbot = interpolate.interp1d(xbot[::-1],ybot[::-1],kind='cubic')


shape1 = (12,10)
xinterp = np.linspace(0,5,shape1[0])

x1    = np.concatenate((xinterp,xinterp[::-1],[0]))
y1    = np.concatenate((interptop(xinterp),interpbot(xinterp[::-1])[::-1],[4]))
beta1 = np.concatenate(([1],np.zeros(shape1[0]-2),[1,1],np.zeros(shape1[0]-2),[1,0]))



# Création de la grille + extraction des données de grille : 
grid = pygridgen.Gridgen(x1, y1, beta1, shape=shape1)
maps = create_maps(grid.x,grid.y)
mapping = get_mapping(maps, grid.x, grid.y)
shapes = extract_shapes(maps)

# Plotting each shapes :
for values in shapes.values() :
    # Getting positions of each nodes for each shapes
    xquad = [mapping[inode][0] for inode in values]
    yquad = [mapping[inode][1] for inode in values]
    # Plotting shapes-frontiers : 
    axes[1].fill(xquad,yquad,
                 color = 'black',
                 linestyle = ':',
                 linewidth = 0.8,
                 facecolor='aliceblue')

# Plotting nodes
axes[1].scatter(grid.x,grid.y,color='black',s= 8)
    
# Plotting each node numbers :
for ix, iy, inode in zip(grid.x_rho.flatten(),
                         grid.y_rho.flatten(),
                         list(shapes.keys())) :
    axes[1].annotate(int(inode),(ix,iy),
                     ha='center',va='center',c='blue',fontsize=6)

# Plotting contours :
axes[1].plot(x1,y1, c='black', label='Contours', linewidth = 1)

# Figure fine-tunning: 
axes[1].set_axisbelow(True)
axes[1].set_title("Grille curvilinéaire\nnon-orthogonale")




# ==== D. Grille curvilinéaire orthogonale ====

xtop = np.array([0,2.5,5])
xbot = np.array([5,2.5,0])
ytop = np.array([4.5,5,4.5])
ybot = np.array([0.5,0,0.5])
interptop = CubicSpline(xtop,      ytop,       bc_type='clamped')
interpbot = CubicSpline(xbot[::-1],ybot[::-1], bc_type='clamped')

shape2 = (8,18)
xinterp = np.linspace(0,5,100)

x2    = np.concatenate((xinterp,xinterp[::-1],[0]))
y2    = np.concatenate((interptop(xinterp),interpbot(xinterp[::-1])[::-1],[4.5]))
beta2 = np.concatenate(([1],np.zeros(100-2),[1,1],np.zeros(100-2),[1,0]))



# Création de la grille + extraction des données de grille : 
grid = pygridgen.Gridgen(x2, y2, beta2, shape=shape2)
maps = create_maps(grid.x,grid.y)
mapping = get_mapping(maps, grid.x, grid.y)
shapes = extract_shapes(maps)

# Plotting each shapes :
for values in shapes.values() :
    # Getting positions of each nodes for each shapes
    xquad = [mapping[inode][0] for inode in values]
    yquad = [mapping[inode][1] for inode in values]
    # Plotting shapes-frontiers : 
    axes[2].fill(xquad,yquad,
                 color = 'black',
                 linestyle = ':',
                 linewidth = 0.8,
                 facecolor='aliceblue')

# Plotting each node numbers :
for ix, iy, inode in zip(grid.x_rho.flatten(),
                         grid.y_rho.flatten(),
                         list(shapes.keys())) :
    axes[2].annotate(int(inode),(ix,iy),
                     ha='center',va='center',c='blue',fontsize=6)

# Plotting contours :
axes[2].plot(x2,y2, c='black', label='Contours', linewidth = 1)

# Plotting wiguedis su'es bords
axes[2].plot([-0.2,0],[0.5,0.5], [0,0],[0.3,0.5],c='red')
axes[2].plot([5,5.2],[0.5,0.5],  [5,5],[0.3,0.5],c='red')
axes[2].plot([-0.2,0],[4.5,4.5], [0,0],[4.5,4.7],c='red')
axes[2].plot([5,5.2],[4.5,4.5],  [5,5],[4.5,4.7],c='red')

# Plotting nodes
axes[2].scatter(grid.x,grid.y,color='black',s= 8)

# Figure fine-tunning: 
axes[2].set_axisbelow(True)
axes[2].set_title("Grille curvilinéaire\northogonale")

for ax,ID in zip(axes,['A','B','C']) :
    ax.text(0.10, 0.95, ID, transform=ax.transAxes,
            fontsize=11, fontweight='bold', va='top', ha='right', color='black')
    ax.set_xticklabels([])
    ax.set_yticklabels([])




fig.tight_layout()
fig.show()
fig.savefig('3_grid_type.png')

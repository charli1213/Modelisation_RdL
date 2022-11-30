import numpy as np

# ====== Gridmaking fonctions  ========

# -------------------------------
"""
def extract_faces(maps) :
#     Extract shapes from a (nx x ny) node ID matrix, since curvilinear.
#    And create à (nx,ny,4) vector which contains the mapping.
    nx,ny = maps.shape
    shape_array = np.empty((nx-1,ny-1,4))
    for j in range(ny-1) :
        for i in range(nx-1) :
            shape_array[i,j,:] = np.array([maps[i,j],
                                           maps[i,j+1],
                                           maps[i+1,j+1],
                                           maps[i+1,j]])
    return shape_array
"""
# -------------------------------
def extract_faces(grid) :
    """ 
    Cette fonction prend une grille et assigne les 'faces' à 4 noeuds.
    Pour se faire, on crée premièrement une 'map' des noeuds, soit une
    matrice ny x nx qui contient la liste des noeuds actif (int) et des
    noeuds inactifs (nan). 

    Ensuite, on crée une 'map' des faces, soit une matrice (ny-1) x (nx-1)
    qui contient 4 noeuds, soit les neouds associés aux faces.
    """
    
    mask = grid.y.mask
    grid_shape = grid.y.mask.shape
    nodes_map = np.empty(grid_shape)

    # --- Création des la 'map' des noeuds :
    id = 0
    for i in range(grid_shape[0]) :
        for j in range(grid_shape[1]) :
            if mask[i,j] == False :
                nodes_map[i,j] = id
                id += 1
            else :
                nodes_map[i,j] = np.nan

    # --- Création des la 'map' des faces :
    ny, nx = grid_shape
    faces_map = np.empty((ny-1,nx-1,4))
    for j in range(ny-1) :
        for i in range(nx-1) :
            faces_map[j,i,:] = np.array([nodes_map[j,i],
                                         nodes_map[j,i+1],
                                         nodes_map[j+1,i+1],
                                         nodes_map[j+1,i]])
            
    # --- On exporte la 'map' des noeuds et la 'map' des faces : 
    return nodes_map, faces_map
    

# -------------------------------


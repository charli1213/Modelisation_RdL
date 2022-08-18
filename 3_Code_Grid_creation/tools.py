import numpy as np

# ====== Gridmaking fonctions  ========
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

def get_mapping(maps, xpos, ypos) :
    """Get you a dictionary containing the position of each nodes (mapping).
    The output has the form {node_number: (xcoord,ycoord)}.
    IN ::
    - maps (np.array) : a mxn matrix containing the id number of each nodes. 
    - xpos, ypos (np.array) : xcoord and ycoord positions of each nodes. 
    OUT :: 
    - mapping (dict) : mapping of the shape {node_number: (xcoord,ycoord)}.
    """
    return {i:(j,k) for i,j,k in zip(maps.flatten(),
                                     xpos.flatten(),
                                     ypos.flatten())}

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

def degree_to_km(deg_array, meanlat) :
    """ Function which convert degree array into kilometer array on 
    the earth, assuming a mean latitude (meanlat [degrees]) for our
    new (x,y) plane. For converting latitude, just assume 
    meanlat = 0 [deg]. 
    (IN) deg_array [np.array] : our vector in degree.
    (IN) meanlat [float] : The mean latitude to evaluate our angles.
    (OUT) km_array [np.array] : The converted vector in km."""
    cearth = 40075 # [KM]
    return deg_array*(np.cos(2*np.pi*(meanlat/360))*(cearth/360))

# -------------------------------

def km_to_degree(km_array, meanlat) :
    """ Function which convert kilometer array into degree array on 
    the earth, assuming a mean latitude (meanlat [degrees]) for our
    new (x,y) plane. For converting latitude, just assume 
    meanlat = 0 [deg].
    (IN) km_array [np.array] : our vector in km.
    (IN) meanlat [float] : The mean latitude to evaluate our angles.
    (OUT) deg_array [np.array] : The converted vector in degrees."""
    cearth = 40075 # [KM]
    return km_array/(np.cos(2*np.pi*(meanlat/360))*(cearth/360))

# -------------------------------

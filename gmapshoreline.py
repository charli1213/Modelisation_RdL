# === Importing modules ===
import numpy as np
import pandas as pd
import xarray as xr
import xugrid as xu
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
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

def get_mapping(maps, xpos, ypos) :
    """Get you a dict with the position of each nodes (mapping)"""
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
    return deg_array*(np.cos(np.pi*(meanlat/360))*(cearth/360))

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
    return km_array/(np.cos(np.pi*(meanlat/360))*(cearth/360))

# -------------------------------













# -------------------------------
# === Parameters ===
midlat = 47.845 # [degrees lat]
nspl = 25 # Number of spline points per curves.
gridshape = (60,40)



# === Opening the CSV file containing Google points (Longitude/Latitude)
datapoint_filename = '/home/charles-edouard/Desktop/TraitementRDL/gridcoastline4.csv'
df = pd.read_csv(datapoint_filename,
                 header=0,
                 index_col=0, sep=",")

# We transform degree lon/lat in meters, because degree lon/lat are not the same degree at that latitude.
df = df.assign(longmeter=degree_to_km(df['longitude'],midlat))
df = df.assign(latmeter=degree_to_km(df['latitude'],0))


# We split the file for each curves (2 curves)
df_rs = df.loc[1:6].sort_values('latmeter') #Rightside
df_ls = df.loc[7:8].sort_values('latmeter') #Leftside


# Les courbes du haut et du bas sont droites. On ajuste les splines latérales pour que ce soit orthogonal : 
bot_dd = -(df.loc[6]['latmeter'] - df.loc[7]['latmeter'])/ \
    (df.loc[6]['longmeter'] - df.loc[7]['longmeter'])
top_dd = -(df.loc[1]['latmeter'] - df.loc[8]['latmeter'])/ \
    (df.loc[1]['longmeter'] - df.loc[8]['longmeter'])


# Creation d'une spline pour les courbes latérales
right_spline = CubicSpline(df_rs['latmeter'], df_rs['longmeter'], bc_type=((1,bot_dd),(1,top_dd)))
left_spline = CubicSpline(df_ls['latmeter'], df_ls['longmeter'], bc_type=((1,bot_dd),(1,top_dd)))
ylat_rs = np.linspace(df_rs['latmeter'].min(),
                      df_rs['latmeter'].max(),
                      nspl)
ylat_ls = np.linspace(df_ls['latmeter'].min(),
                      df_ls['latmeter'].max(),
                      nspl)
rightside = right_spline(ylat_rs)
leftside  = left_spline(ylat_ls)
#plt.plot(leftside,ylat_ls)
#plt.plot(rightside,ylat_rs)


dfgrid = {'longmeter':np.concatenate([rightside,np.flip(leftside)]),
          'latmeter':np.concatenate([ylat_rs,np.flip(ylat_ls)]),
          'beta':np.zeros(2*nspl)}


for i in range(2*nspl) :
    if i in [0,24,25,49] :
        dfgrid['beta'][i] = 1
dfgrid = pd.DataFrame(dfgrid)
dfgrid['longitude'] = km_to_degree(dfgrid.longmeter,midlat)
dfgrid['latitude'] = km_to_degree(dfgrid.latmeter,0)

# === Gridding test === :
foc = pygridgen.grid.Focus()
#foc.add_focus(0.1, axis='x', factor=0.5, extent=0.20)
#foc.add_focus(0.9, axis='x', factor=3.0, extent=0.20)
#foc.add_focus(0.24, axis='y', factor=5.0, extent=0.3) # Verticale à droite 
#foc.add_focus(0.8, axis='y', factor=0.1, extent=0.5) # Verticale à gauche.

grid1 = pygridgen.Gridgen(dfgrid.longitude,
                         dfgrid.latitude,
                         dfgrid.beta,
                         shape=gridshape,
                         focus = foc)
maps = create_maps(grid1.x,grid1.y)
mapping = get_mapping(maps, grid1.x, grid1.y)
shapes = extract_shapes(maps)

# === TRANSFORMATION XUGRID ===
# On crée une bathymétrie aléatoire pour illustrer le processus :

bathymetrie = np.sin(grid1.x)

ds = xr.Dataset({'bathymetrie':(['node'],bathymetrie.flatten()),
                 'latitude':(['node'],grid1.y.flatten()),
                 'longitude':(['node'],grid1.x.flatten()),
                 'face_node_connectivity':
                 (['face','nmax_face'],list(shapes.values())),
                 'mesh2d':([''],[])})
ds.face_node_connectivity.attrs = {'cf_role': 'face_node_connectivity',
                                   'long_name': 'Vertex nodes of mesh faces (counterclockwise)',
                                   'start_index': 0}

ds.mesh2d.attrs = {'cf_role': 'mesh_topology',
                   'long_name': 'Topology data of 2D mesh',
                   'topology_dimension': 2,
                   'node_coordinates': 'longitude latitude',
                   'face_node_connectivity': 'face_node_connectivity',
                   'edge_node_connectivity': 'edge_node_connectivity'}

ds.to_netcdf("xr_grille.nc")
uds = xu.UgridDataset(ds)
bathy = uds.bathymetrie


# === FIGURE CREATION ===
figsize = (7,8)
fig, axes = plt.subplots(2, 2, figsize = figsize)

# --- Top left ---
axes[0,0].scatter(df.longitude,df.latitude,
                  label = 'Échantillon de points',
                  c='blue', s=8, marker = 's')
axes[0,0].fill(dfgrid.longitude, dfgrid.latitude,
               label = 'Contours',
               c = 'black',
               fill = False)


# --- Top Right ---
bathy.ugrid.plot.line(color='black', linewidth=0.4, linestyle='dotted', ax= axes[0,1], label='Grille primaire')
axes[0,1].fill(dfgrid.longitude, dfgrid.latitude,
               label = 'Contours',
               c = 'black',
               fill = False)




# --- Fine tunning : 
for i in [0,1] :
    for j in [0,1] : 
        axes[i,j].set_xlim(-69.60,-69.54)
        axes[i,j].set_ylim(47.82,47.89)
        axes[i,j].set_axisbelow(True)
        fig.tight_layout()
        axes[i,j].legend(loc='best',frameon=False)
        axes[i,j].grid(linestyle=':')

fig.show()

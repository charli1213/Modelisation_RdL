# Ce fichier Python prend en intrant un fichier csv avec les points d'intérêt avec le beta et
# sort deux fichiers. Un netcdf et un csv avec les contours.


# ==== Importing modules ====
# --- Modules mathématiques importants : 
import numpy as np
import math # just for the function "prod"
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import tools as tls

# --- Modules de création des grilles : 
import xarray as xr
import xugrid as xu
import pygridgen

# --- Transformations de coordonnées :
import cartopy.crs as ccrs
import cartopy._epsg as cepsg
from pyproj import Transformer

# --- Progress bars
#from time import sleep
from progress.bar import ChargingBar



# ==== CREATION DES SYSTÈMES DE COORDONNÉES ====
MTM7crs  = cepsg._EPSGProjection(32187)   # Projection MTM7
PlateCarree = ccrs.PlateCarree()         # Projection Mercator
Orthographic = ccrs.Orthographic(-60,47) # Projection Orthographique
transformer_toMTM = Transformer.from_crs(PlateCarree,MTM7crs)



# === Nom des fichiers ===
# --- Directories
input_directory  = '../2_Donnees_entrantes/Polygone_domaine_grille/'
output_directory = 'Grilles_finale/'

# --- Input :
domain_filename = input_directory + 'domaine_grille7.csv'

# --- Output :

gridfile1_path = output_directory  + 'rdl_unrefined_grid_net.nc'
gridfile2_path = output_directory  + 'rdl_refined_grid_net.nc'
nouveau_domaine = output_directory + 'nouveau_domaine_grille.csv'



# === GRIDMAKING ===
# --- Parametres physiques ---
midlat = 47.87 # [degrees lat]
nspl = 25 # Number of spline points per curves.
ny1 = 40
nx1 = 70
gridshape1 = (ny1,nx1)
ny2 = 270
nx2 = 600
gridshape2 = (ny2,nx2)


# --- On ouvre le CSV contenant le domaine de notre grille (lon/lat/beta)
print('\n')
print('Ouverture du domaine de la future grille au')
print('"{}"\n'.format(domain_filename))

df = pd.read_csv(domain_filename,
                 index_col=0,
                 sep=",")


# --- On force l'orthogonalité aux coins ::
# On force une conversion de degrée lon/lat vers un plan x-y.
# Tranformation des limites en MTM7
xmtm7,ymtm7 = transformer_toMTM.transform(df.longitude,df.latitude)
df = df.assign(xmtm7 = xmtm7)
df = df.assign(ymtm7 = ymtm7)


# --- On sépare le dataframe de sorte à avoir séparément toutes
# les "cotés" du polygone format notre domaine.
cutlist, = np.where(df.beta!=0)
cutlist += 1
ncurves = len(cutlist)
slicelist = [slice(cutlist[i],cutlist[i+1]) for i in range(ncurves-1)]
Curves = {i:df.loc[sli] for sli,i in \
               zip(slicelist,range(ncurves-1))}
Curves[ncurves-1] = pd.concat([df.loc[len(df):],df.loc[:1]]).set_index([pd.Index([0,1])])


# --- On trouve la dérivée perpendiculaire aux coins du domaine 
dxdy = {} # Dérivée Orthogonale dx/dy
for i in range(ncurves) :
    Curves[i] = Curves[i].sort_values('ymtm7'*(i%2==0) or 'xmtm7')
    
    # On trouve les dérivées dy/dx pour toutes les courbes, mais
    # on a seulement besoin des latérales.
    dxdy[i] = -( (Curves[i].iloc[ 0]['ymtm7'] \
                  -Curves[i].iloc[-1]['ymtm7'] ) \
                 /( Curves[i].iloc[ 0]['xmtm7'] \
                    -Curves[i].iloc[-1]['xmtm7']) )
    
# --- On crée des splines ORTHOGONALES pour encadrer notre
#     domaine à l'aide des dérivées dxdy. 
Splines = {}
YLat = {}
Direction = {}
for i in range(0,ncurves,2) :
    enddydx = ((1,dxdy[(i-1)%ncurves]),
               (1,dxdy[(i+1)%ncurves]))
    YLat[i] = np.linspace(Curves[i]['ymtm7'].min(),
                       Curves[i]['ymtm7'].max(),
                       nspl)
    Direction[i] = (Curves[i].latitude.iloc[0] == \
                    Curves[i].latitude.sort_index().iloc[0])
    if not Direction[i] :
        enddydx = tuple(reversed(enddydx)) # Orientation des dxdy
        YLat[i] = np.flip(YLat[i]) # On flip si mauvaise direction
    Splines[i] = CubicSpline(Curves[i]['ymtm7'],
                             Curves[i]['xmtm7'],
                             bc_type=enddydx)
    

# --- On reconstruit le dataframe décrivant le domaine de notre grille,
#     mais avec des courbes orthogonales.
dfgrid = {'xmtm7':
          np.concatenate([Splines[i](YLat[i]) for i in range(0,ncurves,2)]),
          'ymtm7':
          np.concatenate([YLat[i] for i in range(0,ncurves,2)]),
          'beta':np.zeros(len(YLat)*nspl)}

# On réattribue les bons beta : 
ibeta = np.sort(np.array([[i*nspl,(i*nspl-1)%(len(YLat)*nspl)] \
                          for i in range(int(ncurves/2))]).flatten())
dfgrid['beta'][ibeta] = df.iloc[np.where(df.beta!=0)].beta.values

# --- On crée finalement le Pandas Dataframe des splines
dfgrid = pd.DataFrame(dfgrid)


# === Gridding === :
# Adding focus
foc = pygridgen.grid.Focus()
#foc.add_focus(0.56, axis='y', factor=25.0, extent=0.60)#
#foc.add_focus(0.60, axis='x', factor=30.0, extent=0.60)
foc.add_focus(0.5975, axis='y', factor=30.0, extent=0.50)#
foc.add_focus(0.60, axis='x', factor=25.0, extent=0.50)

# Y'a plus de points en x.

# Creating grid
bar = ChargingBar('Création de la grille avec PyGridGen en cours...', max = 4)
grid1 = pygridgen.Gridgen(dfgrid.xmtm7,
                          dfgrid.ymtm7,
                          dfgrid.beta,
                          shape=gridshape1)

grid2 = pygridgen.Gridgen(dfgrid.xmtm7,
                          dfgrid.ymtm7,
                          dfgrid.beta,
                          shape=gridshape2,
                          focus = foc)
bar.next()


# ==== CONSTRUCTION DES DATASET.XARRAY ====
# --- Quelques variables à tester.
bathymetrie1 = np.zeros(grid1.shape)
bathymetrie2 = np.zeros(grid2.shape)
distance1 = np.sqrt(grid1.dx.data**2 + grid1.dy.data**2)
distance2 = np.sqrt(grid2.dx.data**2 + grid2.dy.data**2)
bathymetrie1[:-1,:-1] = distance1
bathymetrie2[:-1,:-1] = distance2


# --- Grille de base :
# NOTE : Pas besoin de flatten quand tu appliques un masque qui coupe.
nodes_map1, faces_map1 = tls.extract_faces(grid1)
nodes_mask1  = ~grid1.y.mask.astype('bool')
faces_mask1 = grid1.mask.astype('bool')
faces_IDs1  = np.array(range(len(faces_map1[faces_mask1])))
node_IDs1    = nodes_map1[nodes_mask1]
#
ds1 = xr.Dataset({'bathymetrie':(['node'],bathymetrie1[nodes_mask1]),
                  'face_node_connectivity': (['face','nmax_face'],
                                             faces_map1[faces_mask1]),
                  'mesh2d':([''],[]),
                  'distance':(['face'],distance1[faces_mask1])
                  },
                 coords={
                     "node":(['node'], node_IDs1),
                     "face":(['face'], faces_IDs1),
                     "nmax_face":(['nmax_face'],np.array(range(4)) ),
                     'xmtm7':(['node'],grid1.x.data[nodes_mask1]),
                     'ymtm7':(['node'],grid1.y.data[nodes_mask1])
                 })
bar.next()


# --- Grille raffinée : 
nodes_map2, faces_map2 = tls.extract_faces(grid2)
nodes_mask2  = ~grid2.y.mask.astype('bool')
faces_mask2 = grid2.mask.astype('bool')
faces_IDs2  = np.array(range(len(faces_map2[faces_mask2])))
node_IDs2    = nodes_map2[nodes_mask2]
#
ds2 = xr.Dataset({'bathymetrie':(['node'],bathymetrie2[nodes_mask2]),
                  'face_node_connectivity': (['face','nmax_face'],
                                             faces_map2[faces_mask2]),
                  'mesh2d':([''],[]),
                  'distance':(['face'],distance2[faces_mask2])
                  },
                 coords={
                     "node":(['node'], node_IDs2),
                     "face":(['face'], faces_IDs2),
                     "nmax_face":(['nmax_face'],np.array(range(4)) ),
                     'xmtm7':(['node'],grid2.x.data[nodes_mask2]),
                     'ymtm7':(['node'],grid2.y.data[nodes_mask2])
                 })
bar.next()


# --- Ajout des attribus (attrs) de liés à notre topologie :
facenode_attrs = {'cf_role': 'face_node_connectivity',
                  'long_name':
                  'Vertex nodes of mesh faces (counterclockwise)',
                  'start_index': 0}

mesh2d_attrs = {'cf_role': 'mesh_topology',
                'long_name': 'Topology data of 2D mesh',
                'topology_dimension': 2,
                'node_coordinates': 'xmtm7 ymtm7',
                'face_node_connectivity': 'face_node_connectivity'}
xcoord_attrs = {'standard_name':'projection_x_coordinate',
                'long_name':'x coordinates in NAD83/MTM zone 7',
                'units':'m'}
ycoord_attrs = {'standard_name':'projection_y_coordinate',
                'long_name':'y coordinates in NAD83/MTM zone 7',
                'units':'m'}
               


ds2['distance'].attrs = {'long_name':r'Densité grille ($\sqrt{dx^2+dy^2}$)', 'units':'m'}
ds1.face_node_connectivity.attrs = facenode_attrs
ds2.face_node_connectivity.attrs = facenode_attrs
ds1.xmtm7.attrs = xcoord_attrs
ds2.xmtm7.attrs = xcoord_attrs
ds1.ymtm7.attrs = ycoord_attrs
ds2.ymtm7.attrs = ycoord_attrs
ds1.mesh2d.attrs = mesh2d_attrs
ds2.mesh2d.attrs = mesh2d_attrs
bar.next()
bar.finish()


# --- Sauvegarde au format NetCDF et CSV :
print('\n')
pprint('Paramètre de la grille raffinée ::')
print('--------------------------------::')
print('min(dx) = {} m'.format((grid2.dx).min()))
print('min(dy) = {} m'.format((grid2.dy).min()))
print('--------------------------------::\n')
print('Écriture du domaine de la grille raffinée au')
print('"{}"\n'.format(nouveau_domaine))
dfgrid.to_csv(nouveau_domaine)
print('Sauvegarde de la grille non-raffinée au')
print('"{}"\n'.format(gridfile1_path))
ds1.to_netcdf(gridfile1_path)
print('Sauvegarde de la grille raffinée au')
print('"{}"\n'.format(gridfile2_path))
ds2.to_netcdf(gridfile2_path)


# --- Ouverture de la grille en figure pour observation. 
uds1 = xu.UgridDataset(ds1)
uds2 = xu.UgridDataset(ds2)

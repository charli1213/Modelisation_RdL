# Ce fichier Python prend en intrant un fichier csv avec les points d'intérêt avec le beta et
# sort deux fichiers. Un netcdf et un csv avec les contours.

# === Importing modules ===
# Modules importants
import numpy as np
import pandas as pd
import xarray as xr
import xugrid as xu
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import pygridgen
import tools as tls

# Progress bars
from time import sleep
from progress.bar import ChargingBar

# === Nom des fichiers ===
# --- Directories
input_directory  = '../2_Donnees_entrantes/Polygone_domaine_grille/'
output_directory = '../5_Donnees_sortantes/'

# --- Input :
domain_filename = input_directory + 'domaine_grille6.csv'

# --- Output :
gridfile1_path = output_directory + 'xr_unrefined_grid.nc'
gridfile2_path = output_directory + 'xr_refined_grid.nc'
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
df = df.assign(longmeter=tls.degree_to_km(df['longitude'],midlat))
df = df.assign(latmeter=tls.degree_to_km(df['latitude'],0))


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
    Curves[i] = Curves[i].sort_values('latmeter'*(i%2==0) or 'longmeter')
    
    # On trouve les dérivées dy/dx pour toutes les courbes, mais
    # on a seulement besoin des latérales.
    dxdy[i] = -( (Curves[i].iloc[ 0]['latmeter'] \
                  -Curves[i].iloc[-1]['latmeter'] ) \
                 /( Curves[i].iloc[ 0]['longmeter'] \
                    -Curves[i].iloc[-1]['longmeter']) )
    
# --- On crée des splines ORTHOGONALES pour encadrer notre
# domaine à l'aide des dérivées dxdy. 
Splines = {}
YLat = {}
Direction = {}
for i in range(0,ncurves,2) :
    enddydx = ((1,dxdy[(i-1)%ncurves]),
               (1,dxdy[(i+1)%ncurves]))
    YLat[i] = np.linspace(Curves[i]['latmeter'].min(),
                       Curves[i]['latmeter'].max(),
                       nspl)
    Direction[i] = (Curves[i].latitude.iloc[0] == \
                    Curves[i].latitude.sort_index().iloc[0])
    if not Direction[i] :
        enddydx = tuple(reversed(enddydx)) # Orientation des dxdy
        YLat[i] = np.flip(YLat[i]) # On flip si mauvaise direction
    Splines[i] = CubicSpline(Curves[i]['latmeter'],
                             Curves[i]['longmeter'],
                             bc_type=enddydx)
    

# --- On reconstruit le dataframe décrivant le domaine de notre grille,
# mais avec des courbes orthogonales.
dfgrid = {'longmeter':
          np.concatenate([Splines[i](YLat[i]) for i in range(0,ncurves,2)]),
          'latmeter':
          np.concatenate([YLat[i] for i in range(0,ncurves,2)]),
          'beta':np.zeros(len(YLat)*nspl)}

# On réattribue les bons beta : 
ibeta = np.sort(np.array([[i*nspl,(i*nspl-1)%(len(YLat)*nspl)] \
                          for i in range(int(ncurves/2))]).flatten())
dfgrid['beta'][ibeta] = df.iloc[np.where(df.beta!=0)].beta.values

# --- On crée finalement le Pandas Dataframe
dfgrid = pd.DataFrame(dfgrid)
dfgrid['longitude'] = tls.km_to_degree(dfgrid.longmeter,midlat)
dfgrid['latitude'] = tls.km_to_degree(dfgrid.latmeter,0)


# === Gridding === :
# Adding focus
foc = pygridgen.grid.Focus()
#foc.add_focus(0.10, axis='y', factor=0.3, extent=0.15)
foc.add_focus(0.58, axis='y', factor=25.0, extent=0.60)
foc.add_focus(0.60, axis='x', factor=30.0, extent=0.60)
# Y'a plus de points en x.

# Creating grid
bar = ChargingBar('Création de la grille avec PyGridGen en cours...', max = 5)
grid1 = pygridgen.Gridgen(dfgrid.longmeter,
                          dfgrid.latmeter,
                          dfgrid.beta,
                          shape=gridshape1)
bar.next()
grid2 = pygridgen.Gridgen(dfgrid.longmeter,
                          dfgrid.latmeter,
                          dfgrid.beta,
                          shape=gridshape2,
                          focus = foc)
bar.next()


# Après avoir construit la grille en km, on la retransfère en
# coordonnées lon/lat.
gridlon1 = tls.km_to_degree(grid1.x, midlat)
gridlat1 = tls.km_to_degree(grid1.y, 0)
gridlon2 = tls.km_to_degree(grid2.x, midlat)
gridlat2 = tls.km_to_degree(grid2.y, 0)

bar.next()
# On trouve les shapes et le mapping dans el but de créer la topologie.
maps1 = tls.create_maps(gridlon1,gridlat1)
maps2 = tls.create_maps(gridlon2,gridlat2)
mapping1 = tls.get_mapping(maps1, gridlon1, gridlat1)
mapping2 = tls.get_mapping(maps2, gridlon2, gridlat2)
shapes1 = tls.extract_shapes(maps1)
shapes2 = tls.extract_shapes(maps2)

# === Sauvegarde NetCDF ===
bathymetrie1 = np.sin(grid1.x)
bathymetrie2 = np.sin(grid2.x)
ds1 = xr.Dataset({'bathymetrie':(['node'],bathymetrie1.flatten()),
                  'face_node_connectivity': (['face','nmax_face'],
                                             list(shapes1.values())),
                  'mesh2d':([''],[])
                  },
                 coords={
                     "node":(['node'],maps1.flatten()),
                     "face":(['face'],list(shapes1.keys())),
                     "nmax_face":(['nmax_face'],range(4)),
                     'latitude':(['node'],gridlat1.flatten()),
                     'longitude':(['node'],gridlon1.flatten())
                 })

ds2 = xr.Dataset({'bathymetrie':(['node'],bathymetrie2.flatten()),
                 'face_node_connectivity': (['face','nmax_face'], list(shapes2.values())),
                 'mesh2d':([''],[])
                 },
                coords={
                    "node":(['node'],maps2.flatten()),
                    "face":(['face'],list(shapes2.keys())),
                    "nmax_face":(['nmax_face'],range(4)),
                    'latitude':(['node'],gridlat2.flatten()),
                    'longitude':(['node'],gridlon2.flatten())
                })
bar.next()

# Attribus (attrs) de notre topologie :
facenode_attrs = {'cf_role': 'face_node_connectivity',
                  'long_name':
                  'Vertex nodes of mesh faces (counterclockwise)',
                  'start_index': 0}
mesh2d_attrs = {'cf_role': 'mesh_topology',
                   'long_name': 'Topology data of 2D mesh',
                   'topology_dimension': 2,
                   'node_coordinates': 'longitude latitude',
                   'face_node_connectivity': 'face_node_connectivity',
                   'edge_node_connectivity': 'edge_node_connectivity'}

ds1.face_node_connectivity.attrs = facenode_attrs
ds2.face_node_connectivity.attrs = facenode_attrs
ds1.mesh2d.attrs = mesh2d_attrs
ds2.mesh2d.attrs = mesh2d_attrs
bar.next()
bar.finish()

# Sauvegarde au format NetCDF et CSV :
print('\n')
print('Paramètre de la grille raffinée ::')
print('--------------------------------::')
print('min(dx) = {} m'.format((grid2.dx*1000).min()))
print('min(dy) = {} m'.format((grid2.dy*1000).min()))
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

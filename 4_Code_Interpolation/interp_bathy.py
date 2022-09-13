# ==== IMPORTATION DES MODULES ====
# --- Modules de base
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import xarray as xr
from scipy.interpolate import griddata
# --- Importation pour les cartes
import cartopy.crs as ccrs
import cartopy._epsg as cepsg
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import pandas as pd
from pylab import *
# --- Transformations de coordonnées :
from pyproj import Transformer
# --- Importation de GeoPandas pour les cartes :
from shapely.geometry import Point, Polygon
import geopandas as gpd



# ==== CREATION DES SYSTÈMES DE COORDONNÉES ====
MTM7crs  = cepsg._EPSGProjection(32187)   # Projection MTM7
UTM19crs = cepsg._EPSGProjection(32619)   # Projection UTM19N
PlateCarree = ccrs.PlateCarree()         # Projection Mercator
Orthographic = ccrs.Orthographic(-60,47) # Projection Orthographique
transformer1 = Transformer.from_crs(PlateCarree,MTM7crs)
transformer2 = Transformer.from_crs(UTM19crs,MTM7crs)



# ==== CRÉATION GRILLE SOUS-JACENTE À L'INTERPOLATION ====
# 1. Limites : 
lonmin = -69.58
lonmax = -69.545
latmin = 47.82
latmax = 47.86
extent = [lonmin, lonmax, latmin, latmax]
# Limites en mtm7
xmin,ymin = transformer1.transform(lonmin,latmin)
xmax,ymax = transformer1.transform(lonmax,latmax)

# 2. Creation d'une meshgrid général pour l'interpolation.
nxinterp = 1000
nyinterp = 2000
xdomain = np.linspace(xmin,xmax,nxinterp)
ydomain = np.linspace(ymin,ymax,nyinterp)
X,Y = np.meshgrid(xdomain,ydomain,indexing='ij')




# ==== OUVERTURES DES DONNÉES TOPOBATHYMÉTRIQUES ====
# 1. On ouvre la topobathy de drones sous forme NetCDF (1/10 de résolution)
print('1 :: Ouverture Topobathy par drones')
ds = xr.open_dataset('../5_Donnees_sortantes/topobathy_drone.nc')
xlen = len(ds.x)
ylen = len(ds.y)
nz = 15
temp_ds = ds.isel(x=slice(0,xlen,nz),
                  y=slice(0,ylen,nz)).sel(band=1).rename({'band_data':'CGVD28'})
del temp_ds['band']
spatial_ref = ds.spatial_ref # Pour la création du netcdf
del ds 
drone_da = temp_ds.CGVD28.rename({'x':'Easting',
                                  'y':'Northing'}).stack(ID=('Easting','Northing'))
drone_da = drone_da.dropna('ID')


# 2.a) On ouvre les données de l'hydrobole sous forme CSV
print("2 :: Ouverture données de l'hydrobole")
hydro_filename = '../2_Donnees_entrantes/Hydrolball/RDL_HYDROBALL_MTM7_CGVD28_FZ.csv'
hydro_ds = pd.read_csv(hydro_filename,
                       header=0,
                       index_col=0,
                       sep=",").to_xarray().set_coords(['latitude',
                                                        'longitude',
                                                        'Easting',
                                                        'Northing'])
hydro_ds = hydro_ds.set_index(station = ['Easting','Northing'])
hydro_ds = hydro_ds.rename_dims({'station':'ID'})
hydro_ds = hydro_ds.rename({'station':'ID'})
hydro_ds = hydro_ds.rename({'height (CGVD28)':'CGVD28'})
hydro_da = hydro_ds['CGVD28']

# 2.b) On limite les frontières de l'hydrobole (parce que)
hydro_da = hydro_da.where(hydro_da.longitude<lonmax).where(hydro_da.longitude>lonmin)
hydro_da = hydro_da.where(hydro_da.latitude<latmax ).where(hydro_da.latitude>latmin).dropna('ID').drop(['longitude','latitude'])


# 3.a) On ouvre les données du NONNA :
print('3 :: Ouverture des données du NONNA')
nonna_filename = '../2_Donnees_entrantes/NONNA/RDL_NONNA10_MTM7_CGVD28_FZ.csv'
nonna_ds = pd.read_csv(nonna_filename,
                       header=0,
                       index_col=0,
                       sep=";").to_xarray().set_coords(['latitude',
                                                        'longitude',
                                                        'Easting',
                                                        'Northing'])
nonna_ds = nonna_ds.set_index(ID = ['Easting','Northing'])
nonna_ds['CGVD28'] = nonna_ds['Depth CGVD28']
nonna_da = nonna_ds['CGVD28']

# 3.b) On limite les frontières du Nonna (parce que)
nonna_da = nonna_da.where(nonna_da.longitude<-69.555).where(nonna_da.longitude>lonmin)
nonna_da = nonna_da.where(nonna_da.latitude<latmax ).where(nonna_da.latitude>47.8275).dropna('ID').drop(['longitude','latitude'])


# 4. Ouverture du CSV contenant le trait de côte basse résolution de la
#    pointe de Rivière-du-Loup.
print('4 :: Processing de la pointe basse résolution')
pointe_path = '/home/charles-edouard/Desktop/Traitement_RdL/2_Donnees_entrantes/Trait_de_cote_pointe_RdL/'
pointe_filename = pointe_path + 'trait_de_cote_pointe_rdl_lowres.csv'
dfpointe = pd.read_csv(pointe_filename, sep=",")

# 4.a) Transformation Lon/Lat --> mtm7 pour le trait de côte de la pointe de RdL.
xmtm7, ymtm7 = transformer1.transform(dfpointe.longitude,
                                      dfpointe.latitude)
pointe = matplotlib.path.Path([(x,y) for x,y in zip(xmtm7,ymtm7)])
meshpoints = [(x,y) for x,y in zip(X.flatten(),Y.flatten())]
flags = pointe.contains_points(meshpoints)

# 4.b) Création d'un DataArray correspondant à la topo de la pointe de RdL.
pointe_da = xr.DataArray(flags,
                         coords = {'Easting':('ID',X.flatten()),
                                   'Northing':('ID',Y.flatten())},
                         dims   = ['ID'],
                         attrs  = {'name':'CGVD28'}).set_index(ID = ['Easting','Northing'])
pointe_da = pointe_da.where(pointe_da>0).dropna('ID')*10.


# 5. Ouvertude des données multifaisceaux (Fonction)
# --------------------
print('5 :: Ouverture des données multifaisceaux')
def process_multifaisceaux(filename,dataname,res=1) :
    """Ouvre les données de multifaisceaux en format netcdf et les transforme en
    Xarray.DataArray avec le bon système de coordonnées mtm7.
    filename (str) : Fichier à ouvrir.
    dataname (str) : Nom des données à ouvrir dans le NetCDF.
    res (int)      : Résolution : 1=1, 2=1/2, ...
    exit_da (Xarray.DataArray) : Données sortantes en MTM7.
"""
    exit_da = xr.open_dataset(filename)[dataname].drop(['lon','lat'])
    xlen, ylen = shape(exit_da)
    exit_da = exit_da.isel(x=slice(0,xlen,res),
                           y=slice(0,ylen,res))
    exit_da = exit_da.stack(ID=('x','y')).dropna('ID').rename({'x':'Easting','y':'Northing'})
#    xmtm7, ymtm7 = transformer2.transform(exit_da.x,
#                                          exit_da.y)
#    exit_da = exit_da.assign_coords({'Easting' :('ID',xmtm7),
#                                     'Northing':('ID',ymtm7)})
#    exit_da = exit_da.set_index(ID=('Easting','Northing')).drop(['x','y'])
    return exit_da
# --------------------


# 5.a) Ouverture des données multifaisceaux (Rivière)
filename = '/home/charles-edouard/Desktop/Traitement_RdL/2_Donnees_entrantes/Multifaisceaux/' + 'riviere_25cm_CSRS_MTM7_HT20.nc'
dataname = 'riviere_25cm_CSRS_MTM7_HT20'
riviere_da = process_multifaisceaux(filename,dataname,1)

# 5.b) Ouverture des données multifaisceaux (Large)
filename = '/home/charles-edouard/Desktop/Traitement_RdL/2_Donnees_entrantes/Multifaisceaux/' + 'explo_large_25cm_CSRS_MTM7_HT20.nc'
dataname = 'explo_large_25cm_CSRS_MTM7_HT20'
large_da = process_multifaisceaux(filename,dataname,2)

# 5.c) Ouverture des données multifaisceaux (baie)
filename = '/home/charles-edouard/Desktop/Traitement_RdL/2_Donnees_entrantes/Multifaisceaux/' + 'aoi_50cm_CSRS_MTM7_HT20.nc'
dataname = 'aoi_50cm_CSRS_MTM7_HT20'
baie_da = process_multifaisceaux(filename,dataname,2)


# 6. On fusionne finalement toutes les DataArray ensemble.
print('6 :: Fusion des données pré-interpolation')
da = xr.concat([hydro_da,
                drone_da,
                nonna_da,
                large_da,
                pointe_da,
                riviere_da,
                baie_da],'ID').dropna('ID')



# ==== INTERPOLATION ====
print('INTERPOLATION')
gridz = griddata(list(zip(da.Easting.values,
                          da.Northing.values)),
                 da.values, (X,Y),
                 method='linear')



# ==== CREATION DE LA FIGURE ====
print('CREATION FIGURE')
marker_size = 4
subdivision = 14
figsize = (14,8)
fig,axes = plt.subplots(1,2,figsize=figsize,
                        subplot_kw=dict(projection = Orthographic))
gl0 = axes[0].gridlines(draw_labels=True, linestyle = ':',color='k')
gl1 = axes[1].gridlines(draw_labels=True, linestyle = ':',color='k')
gl0.top_labels = gl0.right_labels = False
gl1.top_labels = gl1.right_labels = False
#cmap = cm.get_cmap('RdYlBu_r',subdivision) # Colormap
cmap = cm.get_cmap('RdBu_r',subdivision) # Colormap
#cmap = cm.get_cmap('jet',subdivision) # Colormap


# Colormap limits : 
absval = max(abs(da.min()),abs(da.max()))
absval = max(-15,15)
# 1..) Gmap
import cartopy.io.img_tiles as cimgt
request1 = cimgt.GoogleTiles(style='street',desired_tile_form='RGBA')
request2 = cimgt.GoogleTiles(style='satellite')
axes[0].add_image(request1, 16, alpha=0.5)
axes[1].add_image(request2, 16, alpha=0.4)

# 1.a) Drone :
axes[0].scatter(drone_da.Easting,
                drone_da.Northing,
                c = 'teal', s = marker_size,
                label = 'Topobathymétrie drone',
                transform = MTM7crs)

# 1.b) NONNA :
axes[0].scatter(nonna_da.Easting,
                nonna_da.Northing,
                c = 'darkred', s = marker_size,
                label = 'Données Nonna',
                transform = MTM7crs)

# 1.c) Pointe :
axes[0].scatter(pointe_da.Easting,
                pointe_da.Northing,
                c = 'goldenrod', s = marker_size,
                label = 'Pointe de RdL (données fictives)',
                transform = MTM7crs)

# 1.f) Large multifaisceaux :
axes[0].scatter(large_da.Easting,
                large_da.Northing,
                c = 'red', s = marker_size,
                label = 'Données multifaisceaux (Large)',
                transform = MTM7crs)

# 1.g) Baie multifaisceaux :
axes[0].scatter(baie_da.Easting,
                baie_da.Northing,
                c = 'MidnightBlue', s = marker_size,
                label = 'Données multifaisceaux (Baie)',
                transform = MTM7crs)
# 1.d) Hydro : 
axes[0].scatter(hydro_da.Easting,
                hydro_da.Northing,
                c = 'greenyellow', s = marker_size,
                label = 'Données hydrobole',
                transform = MTM7crs)
# 1.e) Rivière multifaisceaux :
axes[0].scatter(riviere_da.Easting,
                riviere_da.Northing,
                c = 'orange', s = marker_size,
                label = 'Données multifaisceaux (Rivière)',
                transform = MTM7crs)




# 2. Plotting grid interpolation : 
im1 = axes[1].contourf(X,Y,gridz,subdivision,
                       cmap=cmap,
                       transform = MTM7crs,
                       vmin = -absval, vmax=absval)
axes[1].contour(X,Y,gridz,subdivision,
                linewidths=0.3,colors='k',
                transform = MTM7crs)
cbar = fig.colorbar(cm.ScalarMappable(norm=im1.norm, cmap=im1.cmap), ax=axes[1],extend='both')


# Plotting/filling polygone de la pointe de RdL:
#axes[1].fill(pointe.vertices[:,0],
#             pointe.vertices[:,1],
#             transform=MTM7crs)


# Fine tunning
# Colorbar


cbar.ax.set_ylabel('Altitude [m]', rotation=270)
for axe in axes :
    axe.set_extent(extent)
axes[0].set_title('Illustration des données existantes')
axes[1].set_title('Données interpolées')
axes[0].legend(loc='lower right')
fig.tight_layout()
plt.show()


# ==== SAVING DATA IN NETCDF FORMAT ====
topobathy_da = xr.DataArray(gridz,
                            coords = {'xmtm7':xdomain,
                                      'ymtm7':ydomain},
                            dims = ['xmtm7','ymtm7'],
                            attrs = {'name':'topobathy',
                                     'long_name':'Topobathymétrie RdL',
                                     'units':'meters (m)',
                                     'processus':"Interpolation linéaire depuis Nonna, Drone, Hydrobole et radar multifaisceau. La pointe de RdL est remplie de valeurs ficitve pour améliorer l'interpolation",
                                     'coords_system':'MTM7 (NAD83[CSRS])',
                                     'ellipsoide':'CGVD28'})

path = '/home/charles-edouard/Desktop/Traitement_RdL/5_Donnees_sortantes/'
netcdf_name = path + 'interpolation_lowres.nc'
xr.Dataset({'topobathy':topobathy_da,
            'spatial_ref':spatial_ref},
           attrs=topobathy_da.attrs).to_netcdf(netcdf_name)




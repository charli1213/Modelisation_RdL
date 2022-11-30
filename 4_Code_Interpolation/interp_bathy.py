# ==== IMPORTATION DES MODULES ====
# --- Modules de base
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.rc('axes', axisbelow=True)
import xarray as xr
from scipy.interpolate import griddata
# --- Importation pour les cartes
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import pandas as pd
from pylab import *
import cartopy.io.img_tiles as cimgt
request = cimgt.GoogleTiles(style='satellite')
# --- Transformations de coordonnées :
import cartopy._epsg as cepsg
from pyproj import Transformer
# --- Importation de GeoPandas pour les cartes :
from shapely.geometry import Point, Polygon, MultiPoint
import geopandas as gpd


# ==== CREATION DES SYSTÈMES DE COORDONNÉES ====
MTM7crs  = cepsg._EPSGProjection(32187)   # Projection MTM7
PlateCarree = ccrs.PlateCarree()         # Projection Mercator
Orthographic = ccrs.Orthographic(-60,47) # Projection Orthographique
transformer_toMTM = Transformer.from_crs(PlateCarree,MTM7crs)



# ==== CRÉATION GRILLE SOUS-JACENTE À L'INTERPOLATION ====
# --- Limites
lonmin = -69.580
lonmax = -69.546
latmin =  47.825
latmax =  47.860
extent = [lonmin, lonmax, latmin, latmax]

# Tranformation des limites en MTM7
xmin,ymin = transformer_toMTM.transform(lonmin,latmin)
xmax,ymax = transformer_toMTM.transform(lonmax,latmax)

# Création meshgrid sous-tendant l'interpolation : 
xdomain = np.linspace(int(xmin),int(xmax),2*(int(xmax)-int(xmin))+1)
ydomain = np.linspace(int(ymin),int(ymax),2*(int(ymax)-int(ymin))+1)
X,Y = np.meshgrid(xdomain,ydomain,indexing='ij')



# ==== FONCTIONS ===
# ---
def sharpening_ds(ds, nz=1) :
    """ 
    Cette fonction prend un xarray.dataset provenant d'un fichier tiff et 
    1. Se débarrasse des données trop précises (nz représente la résolution),
    2. Renomme les coordonnées en MTM7 selon «Easting» et «Northing»,
    3. Retourne un xarray.DataArray dont les coordonnées sont ordonnées
       selon un MultiIndex nommé «ID».
    """
    xlen = len(ds.x)
    ylen = len(ds.y)
    temp_ds = ds.isel(x=slice(0,xlen,nz),
                      y=slice(0,ylen,nz)).sel(band=1).rename({'band_data':'CGVD28'})
    out_da = temp_ds.CGVD28.rename({'x':'Easting',
                                      'y':'Northing'}).stack(ID=('Easting','Northing'))
    out_da = out_da.dropna('ID').drop('band')
    return out_da
# ---

# ---
polygon_path  = "../2_Donnees_entrantes/"

def extract_polygon(da, shapefile = polygon_path + 'Trait_de_cote_pointe_RdL/Polygone_LIDAR_pointe_de_RdL.shp') :
    """
    Cette fonction prend comme entrée un xarray.dataArray et un shapely polygone.
    Le but de cette fontion est d'extraire tous les points à L'EXTÉRIEUR de ce 
    même polygone et de retourner un nouveau dataArray.
    """
    polygon = gpd.read_file(shapefile).geometry[0]
    points  = gpd.GeoSeries([Point(x,y) for x,y in zip(da.Easting.values,da.Northing.values)])
    mask    = points.within(polygon)
    da_out  = da[np.invert(array(mask))]
    return da_out
# ---

# ---
def process_multifaisceaux(filename,dataname,res=1) :
    """
    Ouvre les données de multifaisceaux en format netcdf et les transforme en
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
    return exit_da
# ---


# ---
def index_intersection_dataset(da_fiable, da_biaise, radius=-150) :
    """ 
    Prend 2 dataset et donne les indices associées à l'intersection des deux
    datasets. 
    """
    # Création polygone du périmètre des données fiables :
    im = plt.tricontourf(da_fiable.Easting.values,
                         da_fiable.Northing.values,
                         da_fiable.values*0+1,
                         levels=1)
    xcontour = im.collections[1].get_paths()[0].vertices[:,0]
    ycontour = im.collections[1].get_paths()[0].vertices[:,1]
    polycontour = matplotlib.path.Path([(i[0], i[1]) for i in zip(xcontour,ycontour)])
    plt.close() # Destruction de l'image temporaire.

    # On choisie les données biaisées à l'intérieur du polycontour : 
    xy_intersection = np.array([(x,y) for x,y in zip(da_biaise.Easting.values,
                                                da_biaise.Northing.values)])
    indices = np.where(polycontour.contains_points(xy_intersection, radius=radius))
    xy_intersection = xy_intersection[indices]

    return indices, xy_intersection, xcontour, ycontour
# ---


# ---
def debiaisage(da_fiable, da_biaise,
               method = 'linear',
               cmap='bwr',
               radius = -150,
               res = 0.02,
               texte = 'Biais') :
    """ 
    Cette fonction prend deux Xarray.DataArray et débiaise le second (da_biaise) en
    fonction du premier (da_fiable). Elle retourne la valeur du biais, ainsi qu'une
    figure permettant d'observer la provenance des sources d'erreurs.
    
    La résolution (res) des «bin» de l'histogramme est exprimée en mètres.
    """
    # On obtient les indices et (x,y) de l'intersection de nos deux jeux de données : 
    indices, xy_intersection, xcontour, ycontour = index_intersection_dataset(da_fiable, da_biaise, radius=radius)
    
    # Interpolation des données fiables sur les coordonnées des données biaisées :
    ref_interp = griddata(list(zip(da_fiable.Easting.values,
                                   da_fiable.Northing.values)),
                          da_fiable.values,
                          (xy_intersection[:,0],xy_intersection[:,1]),
                          method=method)

    # Calcul du biais :
    biais = da_biaise[indices] - ref_interp
    biais_moyen = float(np.mean(biais))
    print(texte, 'Moyenne :', biais_moyen)

    # Figure
    fig, axes = plt.subplots(nrows=1, ncols=2,figsize = (10,5.2))
    axes[0].remove()
    axes[0] = fig.add_subplot(1,2, 1, projection=Orthographic, aspect=1)
    gl0 = axes[0].gridlines(draw_labels=True, linestyle = ':',color='k',zorder=1,linewidth=0.4)
    gl0.top_labels = gl0.right_labels = False
    gl0.ylabel_style = {'rotation': 90}

    bin_seq = np.linspace(-0.5,0.5,int(1/res+1))
    n, bin, patches = axes[1].hist(biais, facecolor='orange', edgecolor='white', alpha=0.7,
                                   bins=bin_seq)
    
    mode = round(bin[n.argmax()] + res/2, 3)
    print('Mode :', mode, '({})'.format(str(mode)))
    axes[0].fill(xcontour,ycontour,color='orange',alpha=0.8,label = 'Polygone échantillon\nde référence', transform = MTM7crs,zorder=2)
    image_biais = axes[0].scatter(xy_intersection[:,0],xy_intersection[:,1],
                                  c=biais, cmap=cmap,vmin=-1,vmax=1,s=4, transform=MTM7crs, zorder = 3)
    axes[0].add_image(request, 16, alpha=0.45)
    cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=image_biais.norm, cmap=cmap),
                        ax=axes[0], extend='both',
                        label = "Biais échantillon biaisé (m)")
    [ax.grid() for ax in axes]
    axes[0].set_title("Biais par rapport au jeu de\n données de référence")
    axes[1].set_title('Histograme du biais pour des\nintervalles de {} cm'.format(100*res))
    axes[0].set_xlabel('Easting (m)')
    axes[0].set_ylabel('Northing (m)')
    axes[1].set_xlabel("Biais mesuré dans la zone d'intersection (m)")
    axes[1].set_ylabel('Nombre de récurences pour le biais (-)')
    axes[0].legend(loc='upper left', frameon = False, labelcolor = 'white')
    for axe in axes :
        axe.set_axisbelow(True)
    plt.tight_layout()
    #plt.show()
    plt.close()
    # Fin 
    return mode
# ---




# ==== OUVERTURES DES DONNÉES ====

# 1. On ouvre la topobathy de drones (format NetCDF) (1/10 de résolution) :
print('1 :: Ouverture Topobathy par drones')
#ds = xr.open_dataset('../5_Donnees_sortantes/Netcdf/topobathymetrie_drone.nc')
ds = xr.open_dataset('../2_Donnees_entrantes/Topobathy_drone/Netcdf/topobathymetrie_drone.nc')
drone_spatial_ref = ds.spatial_ref # Utile plus tard pour la création du netcdf.
drone_da = sharpening_ds(ds,nz=10)
del ds


# 2. On ouvre les données de l'hydrobole (format CSV) :
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
hydro_da = hydro_ds['CGVD28'].drop(['longitude','latitude'])
del hydro_ds


# 3 On ouvre les données du NONNA (Format CSV) :
print('3 :: Ouverture des données du NONNA')
# 3.a) Limite du NonNa :
xmin_nona = -69.58
xmax_nona = -69.55
ymin_nona = 47.835
ymax_nona = 47.860

# 3.b) Ouverture des données :
nonna_filename = '../2_Donnees_entrantes/NONNA/RDL_NONNA10_MTM7_CGVD28_FZ.csv'
nonna_ds = pd.read_csv(nonna_filename,
                       header=0,
                       index_col=0,
                       sep=";").to_xarray().set_coords(['latitude','longitude',
                                                        'Easting' ,'Northing'])
nonna_ds = nonna_ds.set_index(ID = ['Easting','Northing'])
nonna_ds['CGVD28'] = nonna_ds['Depth CGVD28']
nonna_da = nonna_ds['CGVD28']
del nonna_ds

# 3.c) Découpage des données :
nonna_da = nonna_da.where(nonna_da.longitude<xmax_nona).where(nonna_da.longitude>xmin_nona)
nonna_da = nonna_da.where(nonna_da.latitude <ymax_nona).where(nonna_da.latitude >ymin_nona)
nonna_da = nonna_da.dropna('ID').drop(['longitude','latitude'])


# 4. Ouverture des données LIDAR (déjà en MTM7) :
print('4 :: Ouverture des données LIDAR')
lidar_path = '../2_Donnees_entrantes/LiDAR_bathymetrique/Netcdf/'
lidar_pointe_file = lidar_path + 'topographie_lidar_pointe_de_RdL.nc'
lidar_cote_s_file = lidar_path + 'topographie_lidar_cote_sud.nc'
ds_lidarp = xr.open_dataset(lidar_pointe_file)
ds_lidarc = xr.open_dataset(lidar_cote_s_file)
da_lidarp = sharpening_ds(ds_lidarp,nz=2)
da_lidarc = sharpening_ds(ds_lidarc,nz=2)
del ds_lidarp, ds_lidarc


# 5. Ouvertude des données multifaisceaux (Fonction)
# 5.a) Ouverture des données multifaisceaux (Rivière)
print('5 :: Ouverture des données multifaisceaux')
filename = '../2_Donnees_entrantes/Multifaisceaux/' + 'riviere_25cm_CSRS_MTM7_HT20.nc'
dataname = 'riviere_25cm_CSRS_MTM7_HT20'
riviere_da = process_multifaisceaux(filename,dataname,1)

"""
# 5.b) Ouverture des données multifaisceaux (Large)
filename = '../2_Donnees_entrantes/Multifaisceaux/' + 'explo_large_25cm_CSRS_MTM7_HT20.nc'
dataname = 'explo_large_25cm_CSRS_MTM7_HT20'
large_da = process_multifaisceaux(filename,dataname,2)
"""

# 5.c) Ouverture des données multifaisceaux (baie)
filename = '../2_Donnees_entrantes/Multifaisceaux/' + 'aoi_50cm_CSRS_MTM7_HT20.nc'
dataname = 'aoi_50cm_CSRS_MTM7_HT20'
baie_da = process_multifaisceaux(filename,dataname,2)




# ==== DÉBIAISAGE ==== :
# --- Calcul du biais (mode)
print('6 :: Calcul du biais')
b_drone_lidar = debiaisage(drone_da,xr.concat([da_lidarc,da_lidarp],'ID'), radius = -50, texte = 'Biais drone/lidar')
b_hydro_drone = debiaisage(drone_da,hydro_da,texte = 'Biais drone/hydro')
b_hydro_baie  = debiaisage(baie_da,hydro_da, texte = 'Biais multibaie/hydro')
b_baie_nonna  = debiaisage(baie_da,nonna_da,radius=-100, texte = 'Biais multibaie/nonna')
b_drone_nonna = debiaisage(drone_da,nonna_da, texte = 'Biais drone/nonna')



# ==== DÉCOUPAGES ==== :
# --- Suppression des données du Nonna aux intersections avec les jeux de données
#     du multifaisceaux et du drone : 
print('Correction NONNA')
index4,xy_intersection, xcontour, ycontour = index_intersection_dataset(baie_da, nonna_da, radius=50)
index5,xy_intersection, xcontour, ycontour = index_intersection_dataset(drone_da, nonna_da, radius=50)
intersection_id = np.sort(np.concatenate([index4[0],index5[0]]))
index_corrige = range(len(nonna_da))
index_corrige = np.delete(index_corrige,intersection_id)
nonna_da = nonna_da[index_corrige]

# --- Découpage des données drone avec la fonction extract_polygon et les
#     shapefile de traits de côte du LIDAR.
print('Découpage des traits de côte LIDAR')
print('Découpage 1')
drone_da = extract_polygon(drone_da, shapefile = polygon_path + 'Trait_de_cote_pointe_RdL/polygone_LiDAR_bathymetrique_pointe_RdL.shp')
print('Découpage 2')
drone_da = extract_polygon(drone_da, shapefile = polygon_path + 'Trait_de_cote_cote_sud/polygone_cote_sud_LiDAR_bathymetrique.shp')




# ==== FUSION FINALE DES DONNÉES PRÉ-INTERPOLATION ====
print('6 :: Fusion des données pré-interpolation')
#nonna_da = nonna_da.drop(['latitude','longitude'])
da = xr.concat([drone_da,
                baie_da - (b_hydro_baie - b_hydro_drone),
                hydro_da - b_hydro_drone,
                da_lidarc - b_drone_lidar,
                da_lidarp - b_drone_lidar,
                nonna_da - b_drone_nonna,
                riviere_da
                ],'ID').dropna('ID')

#                large_da,



# ==== INTERPOLATION FINALE ====
print('INTERPOLATION')
gridz = griddata(list(zip(da.Easting.values,
                          da.Northing.values)),
                 da.values, (X,Y),
                 method='linear')


if __name__ == "__main__":

    # ==== SAVING DATA IN NETCDF FORMAT ====
    topobathy_da = xr.DataArray(gridz,
                                coords = {'xmtm7':xdomain,
                                          'ymtm7':ydomain},
                                dims = ['xmtm7','ymtm7'],
                                attrs = {'name':'topobathy',
                                         'long_name':'Topobathymétrie RdL high resolution',
                                         'units':'meters (m)',
                                         'processus':"Interpolation linéaire high",
                                         'coords_system':'MTM7 (NAD83[CSRS])',
                                         'ellipsoide':'CGVD28'})

    path = 'Topobathymetrie_finale/'
    netcdf_name = path + 'interpolation_highres.nc'
    xr.Dataset({'topobathy':topobathy_da,
                'spatial_ref':drone_spatial_ref},
               attrs=topobathy_da.attrs).to_netcdf(netcdf_name)


    
    

# ==== IMPORTATION DES MODULES ====
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from scipy.interpolate import griddata

# --- Importation pour les cartes
import cartopy.crs as ccrs
import cartopy._epsg as cepsg
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import pandas as pd
from pylab import *

# ==== OUVERTURES DES DONNÉES TOPOBATHYMÉTRIQUES ====
# On ouvre la topobathy de drones sous forme NetCDF
ds = xr.open_dataset('../5_Donnees_sortantes/topobathy_drone.nc')
xlen = len(ds.x)
ylen = len(ds.y)
nz = 10
temp_ds = ds.isel(x=slice(0,xlen,nz),
                  y=slice(0,ylen,nz)).sel(band=1).rename({'band_data':'CGVD28'})
del temp_ds['band']
drone_dA = temp_ds.CGVD28.rename({'x':'Easting',
                                   'y':'Northing'}).stack(ID=('Easting','Northing'))

### Ici ça bugg...


# On ouvre les données de l'hydrobole sous forme CSV
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
hydro_ds['CGVD28'] = hydro_ds['height (CGVD28)']
hydro_dA = hydro_ds['CGVD28']


# On ouvre les données du NONNA :
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
nonna_dA = nonna_ds['CGVD28']

# On fusionne les DataArrays de l'hydrobole et du Nonna : 
da = xr.concat([hydro_dA, nonna_dA], 'ID')

# On limite les frontières de nos données :
lonmin = -69.58
lonmax = -69.55
latmin = 47.82
latmax = 47.86

da = da.where(da.longitude<lonmax).where(da.longitude>lonmin)
da = da.where(da.latitude<latmax).where(da.latitude>latmin).dropna('ID')

# On fusionne finalement toutes les données ensembles.
da = da.drop(['longitude','latitude'])
da = xr.concat([da,drone_dA],'ID').dropna('ID')

# ==== Interpolation ====
# [X] 1. Tout fusionner avec un multiindex 
# [X] - Pas de lon/lat pour le drone...

y = np.linspace(da.Northing.min(),da.Northing.max(),1000)
x = np.linspace(da.Easting.min(),da.Easting.max(),1000)

X,Y = np.meshgrid(x,y)
gridz = griddata(list(zip(da.Easting.values,
                          da.Northing.values)),
                 da.values, (X,Y),
                 method='nearest')


# ==== CREATION DE LA FIGURE ====
figsize = (15,8)
fig,axes = plt.subplots(1,2,figsize=figsize,
                        subplot_kw=dict(projection=ccrs.Orthographic(-60,47)))
crs = cepsg._EPSGProjection(32187) # Projection MTM7  
gl0 = axes[0].gridlines(draw_labels=True, linestyle = ':',color='k')
gl1 = axes[1].gridlines(draw_labels=True, linestyle = ':',color='k')
gl0.top_labels = gl0.right_labels = False
gl1.top_labels = gl1.right_labels = False
cmap = cm.get_cmap('bwr', 20) # Colormap

# Colormap limits : 
absval = max(abs(da.min()),abs(da.max()))

# Plotting/Scatter of all point values :
im0 = axes[0].scatter(da.Easting,
                      da.Northing,
                      c=da.values,
                      transform = crs,
                      vmin = -absval, vmax=absval,
                      cmap=cmap)

# Plotting interpolation : 
im1 = axes[1].pcolormesh(X,Y,gridz,
                         cmap=cmap,
                         transform = crs,
                         vmin = -absval, vmax=absval)

# Fine tunning
# Colorbar
cbar = fig.colorbar(im1, ax=axes[1])
cbar.ax.set_ylabel('Altitude [m]', rotation=270)
fig.tight_layout()
fig.show()

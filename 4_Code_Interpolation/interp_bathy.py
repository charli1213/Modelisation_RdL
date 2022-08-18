# --- Importation des modules
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

# --- Importation pour les cartes
import cartopy.crs as ccrs
import cartopy._epsg as cepsg
from cartopy.feature import NaturalEarthFeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import pandas as pd





# On ouvre la topobathy de drones sous forme NetCDF
ds = xr.open_dataset('../5_Donnees_sortantes/topobathy_drone.nc')
xlen = len(ds.x)
ylen = len(ds.y)
nz = 10
band_data = ds.sel(band=1).isel(x=slice(0,xlen,nz),
                                y=slice(0,ylen,nz)).band_data


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

# On fusionne tous les DataArrays : 
da = xr.concat([hydro_dA, nonna_dA], 'ID')

# Limite des données du NONNA :
lonmin = -69.58
lonmax = -69.55
latmin = 47.82
latmax = 47.86

da = da.where(da.longitude<lonmax).where(da.longitude>lonmin)
da = da.where(da.latitude<latmax).where(da.latitude>latmin).dropna('ID')

# === Creation de la figure : 
fig,axes = plt.subplots(subplot_kw=dict(projection=ccrs.Orthographic(-60,47)))
  
gl = axes.gridlines(draw_labels=True, linestyle = ':',color='k')
gl.top_labels = gl.right_labels = False

# Plotting Drone ::
crs = cepsg._EPSGProjection(32187) # Projection MTM7
absval = max(abs(da.min()),abs(da.max()))
bandplot = band_data.plot(ax=axes, cmap='coolwarm', transform=crs,
                          vmin = -absval, vmax=absval,)


# Plotting Hydrobole et Nonna ::
axes.scatter(da.Easting,
             da.Northing,
             c=da.values,
             transform = crs,
             vmin = -absval, vmax=absval,
             cmap='coolwarm')


# Coastline
coast = NaturalEarthFeature(category='physical', scale='10m',
                            facecolor='none', name='coastline')
axes.add_feature(coast, edgecolor='gray')

plt.tight_layout()
plt.show()

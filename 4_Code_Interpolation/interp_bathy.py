# --- Importation des modules
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr


import cartopy.crs as ccrs
import cartopy._epsg as cepsg
from cartopy.feature import NaturalEarthFeature
import pandas as pd

### Important si le fichier drone.nc a pas été créé ###
#import rioxarray
#filename = 'RDL_20220418_10CM.tif'
#da = xr.open_dataarray(filename, engine = 'rasterio', chunks='auto')
#da.to_netcdf('drone.nc')
### Important si le fichier drone.nc a pas été créé ###




# On ouvre les données de drones
ds = xr.open_dataset('../5_Donnees_sortantes/topobathy_drone.nc')
xlen = len(ds.x)
ylen = len(ds.y)
nz = 10
band_data = ds.sel(band=1).isel(x=slice(0,xlen,nz),
                                y=slice(0,ylen,nz)).band_data


# On ouvre les données dde l'hydrobole :
hydro_filename = '../2_Donnees_entrantes/Hydrolball/RDL_HYDROBALL_MTM7_CGVD28_FZ.csv'
hydro_ds = pd.read_csv(hydro_filename,
                       header=0,
                       index_col=0,
                       sep=",").to_xarray().set_coords(['latitude',
                                                        'longitude',
                                                        'Easting',
                                                        'Northing'])
hydro_ds = hydro_ds.set_index(station = ['longitude','latitude'])
hydro_ds = hydro_ds.set_index(station = ['Easting','Northing'])




from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
fig,axes = plt.subplots(subplot_kw=dict(projection=ccrs.Orthographic(-60,47)))
  
gl = axes.gridlines(draw_labels=True, linestyle = ':',color='k')
gl.top_labels = gl.right_labels = False

# Plotting Drone ::
crs = cepsg._EPSGProjection(32187)
bandplot = band_data.plot(ax=axes, cmap='seismic', transform=crs)
absval = max(abs(band_data.min()),abs(band_data.max()))


# Plotting Hydrobole ::
axes.scatter(hydro_ds.Easting,
             hydro_ds.Northing,
             c=hydro_ds['height (CGVD28)'].values,
             transform = crs,
             vmin = -absval, vmax=absval,
             cmap='seismic')

coast = NaturalEarthFeature(category='physical', scale='10m',
                            facecolor='none', name='coastline')
axes.add_feature(coast, edgecolor='gray')

plt.tight_layout()
plt.show()

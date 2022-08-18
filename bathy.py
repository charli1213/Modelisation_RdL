import matplotlib.pyplot as plt
import xarray as xr
import xugrid as xu
import pandas as pd
import numpy as np

# --- Ouvertude des données.
df1 = pd.read_csv('RDL_HYDROBALL_MTM7_CGVD28_FZ.csv',
                 header=0,
                 index_col=0, sep=",")

df2 = pd.read_csv('RDL_NONNA10_MTM7_CGVD28_FZ.csv',
                  header=0,
                  index_col=0, sep=";")

#df3 = pd.read_csv('/home/charles-edouard/Desktop/TraitementRDL/Coords G-Map/'+'Adresses enregistrées.csv',
#                  header=0,
#                  index_col=0, sep=",")

# --- On transforme latitude et longitude en coordonnées.
ds1 = df1.to_xarray()
ds2 = df2.to_xarray()
ds1 = ds1.set_coords(['latitude','longitude'])
ds2 = ds2.set_coords(['latitude','longitude'])

# --- On attribue un multiindexe à la stations.
ds1 = ds1.set_index(station = ['longitude','latitude'])

#ds1.plot.scatter(x='longitude',y='latitude',z='height (CGVD28)')

#uds1 = xu.UgridDataset(ds1)

xy_low = (47.818299, -69.602377)
xy_upp = (47.859479, -69.544007)

im = plt.imread('rdl90161_109.jpg')
plt.imshow(np.flip(im, axis=0), origin='lower', extent = [xy_low[1],xy_upp[1],
                                                          xy_low[0],xy_upp[0]])
plt.scatter(ds1.longitude, ds1.latitude, c=ds1['height (CGVD28)'], cmap='viridis')
plt.scatter(ds2.longitude, ds2.latitude, c=ds2['Depth CGVD28'], cmap='viridis')
#plt.grid()
#plt.colorbar()
plt.show()



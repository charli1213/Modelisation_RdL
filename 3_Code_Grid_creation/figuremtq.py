import xarray as xr
import xugrid as xu
import pandas as pd
import matplotlib.pyplot as plt

# Opening netcdf with grid.
filename = "xr_grille.nc"
path = '../5_Donnees_sortantes/'
ds1 = xr.open_dataset(path + "xr_unrefined_grid.nc")
ds2 = xr.open_dataset(path + "xr_refined_grid.nc")
uds1 = xu.UgridDataset(ds1)
uds2 = xu.UgridDataset(ds2)


# Ajout des cartes Google aux quadrants 1 et 2
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.io.img_tiles as cimgt




# === CREATION FIGURE  ===
# --- Parameters ---
request = cimgt.GoogleTiles(style='satellite')
#request = cimgt.GoogleTiles()
#request = cimgt.OSM()
figsize = (9,10)
extent = [-69.577, -69.54, 47.832, 47.857]
midlon = extent[1] - extent[0]
midlat =  extent[3] - extent[2]
alpha_grid = 0.6
fig, axes = plt.subplots(2, 2,
                         figsize = figsize,
                         subplot_kw=dict(projection=ccrs.Orthographic(-60,47)))
data_crs = ccrs.PlateCarree()


# --- Ajout Google Map quadrant 1 et 2 :
gl1 = axes[0,0].gridlines(draw_labels=True, linestyle = ':',color='w',alpha=alpha_grid)
gl2 = axes[0,1].gridlines(draw_labels=True, linestyle = ':',color='w',alpha=alpha_grid)
gl3 = axes[1,0].gridlines(draw_labels=True, linestyle = ':',color='w',alpha=alpha_grid)
gl4 = axes[1,1].gridlines(draw_labels=True, linestyle = ':',color='w',alpha=alpha_grid)
gl1.top_labels = gl1.right_labels = gl2.top_labels = gl2.right_labels = False
gl3.top_labels = gl3.right_labels = gl4.top_labels = gl4.right_labels = False
axes[0,0].add_image(request, 16)
axes[0,1].add_image(request, 16)
axes[1,0].add_image(request, 16)

# --- Opening CSV file with sample datapoints :
input_path = '../2_Donnees_entrantes/Polygone_domaine_grille/'
datapoint_filename = input_path + 'domaine_grille6.csv'
df = pd.read_csv(datapoint_filename,
                 header=0,
                 index_col=0, sep=",")
dfgrid = pd.read_csv(path + "nouveau_domaine_grille.csv",
                 header=0,
                 index_col=0, sep=",")

# --- Top left ---
axes[0,0].scatter(df.longitude, df.latitude,
                  label = 'Échantillon de points',
                  c='orange', s=12, marker = 'D',
                  transform=data_crs)

axes[0,0].fill(dfgrid.longitude, dfgrid.latitude,
               label = 'Interpolation Splines',
               c = 'orange',
               fill = False,
               transform=data_crs)


# --- Top Right ---
try :  uds1.face.ugrid.plot.line(color='orange', 
                                linewidth=0.6,
                                ax=axes[0,1],
                                label='Grille primaire orthogonale',
                                transform = data_crs)
except : print('Petit bug, mais ça marche')
    
axes[0,1].set_axisbelow(True)


# --- Bottom left ---
try :  uds2.face.ugrid.plot.line(color='orange', 
                                linewidth=0.6,
                                ax=axes[1,0],
                                label='Grille rafinée',
                                transform = data_crs)
except : print('Petit bug, mais ça marche')


# --- Bottom right ---


# --- Figure Fine Tunning ---
for i in [0,1] :
    for j in [0,1] : 
        axes[i,j].legend(frameon=False, labelcolor = 'white', loc='upper left')
        axes[i,j].set_extent(extent)

plt.tight_layout()
plt.show()


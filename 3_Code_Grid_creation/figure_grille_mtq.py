
# ==== IMPORTATION DES MODULES ====
# --- Modules mathématiques :
import xarray as xr
import xugrid as xu
import pandas as pd
import matplotlib.pyplot as plt
#import matplotlib.cm as cm
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cmocean

# --- Transformations de coordonnées :
import cartopy.crs as ccrs
import cartopy._epsg as cepsg
from pyproj import Transformer

# --- Cartes Google :
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.io.img_tiles as cimgt



# ==== OUVERTURE DES DONNÉES ==== 
# Opening netcdf with grid.
filename = "xr_grille.nc"
path = 'Grilles_finale/'
ds1 = xr.open_dataset(path + "rdl_unrefined_grid_net.nc")
ds2 = xr.open_dataset(path + "rdl_refined_grid_net.nc")
ds2['bathymetrie'].attrs = {'unit':'m', 'long_name':'Distance points de grille'}
uds1 = xu.UgridDataset(ds1)
uds2 = xu.UgridDataset(ds2)






# ==== CREATION DES SYSTÈMES DE COORDONNÉES ====
MTM7crs  = cepsg._EPSGProjection(32187)   # Projection MTM7
PlateCarree = ccrs.PlateCarree()         # Projection Mercator
Orthographic = ccrs.Orthographic(-90,47) # Projection Orthographique
transformer_toMTM = Transformer.from_crs(PlateCarree,MTM7crs)



# ==== CREATION FIGURE  ====
# --- Parameters ---
request = cimgt.GoogleTiles(style='satellite')
figsize = (11,9)
#extent = [-69.57, -69.547, 47.835, 47.857]
extent = [-69.5775, -69.547, 47.835, 47.857]
midlon = extent[1] - extent[0]
midlat =  extent[3] - extent[2]
alpha_grid = 0.6

fig, axes = plt.subplots(nrows=2, ncols=2,
                         figsize = figsize,
                         subplot_kw=dict(projection=Orthographic))



# --- Ajout Google Map quadrant 1 et 2 :
gl1 = axes[0,0].gridlines(draw_labels=True, linestyle = ':',color='w',alpha=alpha_grid,zorder=2)
gl2 = axes[0,1].gridlines(draw_labels=False, linestyle = ':',color='w',alpha=alpha_grid,zorder=2)
gl3 = axes[1,0].gridlines(draw_labels=True, linestyle = ':',color='w',alpha=alpha_grid,zorder=2)
gl4 = axes[1,1].gridlines(draw_labels=True, linestyle = ':',color='w',alpha=alpha_grid,zorder=2)
gl1.top_labels = gl1.right_labels = gl1.bottom_labels = False
#gl2.top_labels = gl2.right_labels = False
gl3.top_labels = gl3.right_labels = False
gl4.right_labels = gl4.top_labels = False
gl1.ylabel_style = {'rotation': 90}
gl2.ylabel_style = {'rotation': 90}
gl3.ylabel_style = {'rotation': 90}
gl4.ylabel_style = {'rotation': 90}
axes[0,0].add_image(request, 16, alpha = 0.82, zorder = 1)
axes[1,0].add_image(request, 16, alpha = 0.82, zorder = 1)
axes[0,1].add_image(request, 16, alpha = 0.82, zorder = 1)
axes[1,1].add_image(request, 18, alpha = 0.82, zorder = 1)

# --- Opening CSV file with sample datapoints :
input_path = '../2_Donnees_entrantes/Polygone_domaine_grille/'
datapoint_filename = input_path + 'domaine_grille7.csv'
df = pd.read_csv(datapoint_filename,
                 header=0,
                 index_col=0, sep=",")
dfgrid = pd.read_csv(path + "nouveau_domaine_grille.csv",
                 header=0,
                 index_col=0, sep=",")

# --- Top left ---
axes[0,0].fill(dfgrid.xmtm7, dfgrid.ymtm7,
               label = 'Splines orthogonales',
               c = 'orange',
               fill = False,
               transform=MTM7crs, zorder = 3)
axes[0,0].scatter(df.longitude, df.latitude,
                label = 'Échantillon de points',
                c='azure', s=12, marker = 'D',
                  transform=PlateCarree, zorder = 4)

# --- Top Right ---
try :  uds1.face.ugrid.plot.line(color='orange', 
                                 linewidth=0.6,
                                 ax=axes[0,1],
                                 label='Grille primaire orthogonale',
                                 transform = MTM7crs,
                                 zorder = 3)
except : print('Petit bug, mais ça marche')
    
axes[0,1].set_axisbelow(True)


# --- Bottom left ---
cmap = cmocean.cm.thermal
im_10 = uds2['distance'].ugrid.plot(ax=axes[1,0],
                                    cmap = cmap,
                                    vmin = 0.5, vmax = 15,
                                    transform = MTM7crs,
                                    zorder = 3,
                                    cbar_kwargs = {'pad':0.018,
                                                   'label':'Séparation points de grille [m]'})
# *** Beaucoup de travail ici pour trouver le "pad" optimal pour que les
#     tableaux soient alignés.


# --- Bottom right ---
try :
    uds2.face.ugrid.plot.line(color='orange', 
                              linewidth=0.6,
                              ax=axes[1,1],
                              label='Grille raffinée',
                              transform = MTM7crs,
                              zorder = 3)
except : print('Petit bug, mais ça marche')



# --- Figure Fine Tunning ---


for ax,ID in zip(axes.flat,['A','B','C','D']) : 
        ax.text(0.05, 0.05, ID, transform=ax.transAxes,
                fontsize=11, fontweight='bold', va='top', ha='right', color='white')
        ax.legend(frameon=False, labelcolor = 'white', loc='upper left')
        ax.set_extent(extent)
        
axes[0,0].sharex(axes[1,0])
        
axes[1,1].set_extent([-69.5541, -69.5525, 47.84107, 47.8423])
axes[0,0].set_title("Génération contours orthogonaux\n")
axes[0,1].set_title("Génération grille primaire orthogonale\n")
axes[1,0].set_title("Grille raffinée\n")
axes[1,1].set_title("Zoom sur le brise-lames")
    
plt.tight_layout()
plt.show()
fig.savefig('../6_Figures_rapport/3_Figures_Grille/figure_finale_grille.png')
plt.close()

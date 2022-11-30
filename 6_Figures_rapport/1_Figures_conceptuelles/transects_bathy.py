# ==== IMPORTATION DES MODULES ====
# --- Modules de base
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
plt.rc('axes', axisbelow=True)
import xarray as xr
import pandas as pd
from scipy.interpolate import griddata
from scipy import interpolate

# --- Importation pour les cartes
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy._epsg as cepsg
from pyproj import Transformer



# ==== CREATION DES SYSTÈMES DE COORDONNÉES ====
MTM7crs  = cepsg._EPSGProjection(32187)   # Projection MTM7
PlateCarree = ccrs.PlateCarree()         # Projection Mercator
Orthographic = ccrs.Orthographic(-60,47) # Projection Orthographique
transformer_toMTM = Transformer.from_crs(PlateCarree,MTM7crs)



# ==== INTERPOLATION DE LA BATHYMETRIE SUR LA GRILLE ====
# --- Ouverture de la topobathymétrie :
path = "../../5_Donnees_sortantes/"
filename = path+"interpolation_finale.nc"
ds = xr.open_dataset(filename)
da = ds['topobathy']
da = da.rename({'xmtm7':'x','ymtm7':'y'})


# --- Transects :
x1 = -69.578669
y1 =  47.83
x2 = -69.555
y2 =  47.827

xmtm70,ymtm70 = transformer_toMTM.transform([x1,x2],[y1,y2])
xvec0 = np.linspace(xmtm70[0], xmtm70[1], 100)
yvec0 = np.linspace(ymtm70[0], ymtm70[1], 100)



x3 = -69.5514
y3 = 47.8323
x4 = -69.5773
y4 = 47.8423

xmtm71,ymtm71 = transformer_toMTM.transform([x3,x4],[y3,y4])
xvec1 = np.linspace(xmtm71[0], xmtm71[1], 100)
yvec1 = np.linspace(ymtm71[0], ymtm71[1], 100)

x5 = -69.5480
y5 = 47.8381
x6 = -69.5753
y6 = 47.8568

xmtm72,ymtm72 = transformer_toMTM.transform([x5,x6],[y5,y6])
xvec2 = np.linspace(xmtm72[0], xmtm72[1], 100)
yvec2 = np.linspace(ymtm72[0], ymtm72[1], 100)


def lookupNearest(da, x, y):
    z_out = []
    for xi,yi in zip(x,y) :
        z_out += [float(da.sel(x=xi,y=yi,method='nearest'))]
    return z_out
z0 = lookupNearest(da, xvec0, yvec0)
z1 = lookupNearest(da, xvec1, yvec1)
z2 = lookupNearest(da, xvec2, yvec2)

# ==== FIGURE ====
figsize = (10,6)
subdivision = 40
c_cmap = 0.2
absval = c_cmap*max(abs(da.min()),abs(da.max()))
cmap = cm.get_cmap('Spectral_r',subdivision) # Coloremap


fig,axes = plt.subplots(ncols = 4, nrows = 3,
                        figsize = figsize,
                        subplot_kw=dict(projection = Orthographic))

gs = axes[0, 0].get_gridspec()

# remove the underlying axes
for ax in axes[:,:].flatten():
    ax.remove()

axbig = fig.add_subplot(gs[:,:2], projection = Orthographic)
axright0 = fig.add_subplot(gs[0,2:])
axright1 = fig.add_subplot(gs[1,2:])
axright2 = fig.add_subplot(gs[2,2:])

gl0 = axbig.gridlines(draw_labels=True, linestyle = ':',color='k')
gl0.top_labels = gl0.right_labels = False
gl0.ylabel_style = {'rotation': 90}



# --- Left plot : 
da.transpose().plot(cmap = cmap,
                    transform = MTM7crs,
                    ax=axbig,
                    add_colorbar = False,
                    vmin = -absval,
                    vmax = absval)
axbig.plot(xvec0,yvec0,c='red',label = 'Transect 1',transform=MTM7crs)
axbig.plot(xvec1,yvec1,c='green',label = 'Transect 2',transform=MTM7crs)
axbig.plot(xvec2,yvec2,c='blue',label = 'Transect 3',transform=MTM7crs)

# --- First right : 
xvec0 = np.linspace(x1,x2,100)
xvec1 = np.linspace(x3,x4,100)
xvec2 = np.linspace(x5,x6,100)
axright0.plot(xvec0,z0,color = 'red')
axright1.plot(xvec1,z1,color = 'green')
axright2.plot(xvec2,z2,color = 'blue')

# --- Fine tunning :
titles = ['Transect 1','Transect 2', 'Transect 3']
axis   = [axright0,axright1,axright2]

for ax,text,ID in zip(axis,titles,['B','C','D']) :
    ax.set_title(text)
    ax.grid()
    ax.set_ylabel('z (m)')
    ax.tick_params(axis='y', labelrotation = 90)

    ax.text(0.05, 0.95, ID, transform=ax.transAxes,
            fontsize=11, fontweight='bold', va='top', ha='right')

axbig.text(0.07, 0.97, 'A', transform=axbig.transAxes,
           fontsize=11, fontweight='bold', va='top', ha='right')
axbig.set_title('Topobathymétrie interpolée')
axbig.legend(loc = 'upper right')

axright2.set_xlabel('Longitude')

fig.tight_layout()
plt.show()

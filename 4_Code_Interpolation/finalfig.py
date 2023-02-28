# ==== IMPORTATION DES MODULES ====
#import cmocean # Pas nécessaire 
import matplotlib.pyplot as plt
import matplotlib as mpl

"""
# ==== CREATION DES SYSTÈMES DE COORDONNÉES ====
MTM7crs  = cepsg._EPSGProjection(32187)   # Projection MTM7
PlateCarree = ccrs.PlateCarree()         # Projection Mercator
Orthographic = ccrs.Orthographic(-60,47) # Projection Orthographique
transformer_toMTM = Transformer.from_crs(PlateCarree,MTM7crs)
"""

# ==== CREATION DE LA FIGURE ====
print('CREATION FIGURE')
subdivision = 40
#figsize = (11,6) #A4
figsize = (18,6.7) #Poster
fig,axes = plt.subplots(1,2,figsize=figsize,
                        subplot_kw=dict(projection = Orthographic))

# --- Gridlines : 
gl0 = axes[0].gridlines(draw_labels=True, linestyle = ':',color='k')
gl1 = axes[1].gridlines(draw_labels=True, linestyle = ':',color='k')
gl0.top_labels = gl0.right_labels = False
gl1.top_labels = gl1.right_labels = False
gl0.ylabel_style = {'rotation': 90}
gl1.ylabel_style = {'rotation': 90}
#extent = [-69.58, -69.545, 47.825, 47.86]

# --- Colormap
cmap = mpl.cm.get_cmap('Spectral_r')

# Get the colormap colors
flat_cmap = cmap(np.arange(cmap.N))

# Change colorbar alpha :
flat_cmap[:,-1] = np.concatenate( (np.ones(int(7*cmap.N/8)),
                                 np.linspace(1, 0.30, int(cmap.N/8))) )

# Create new colormap
flat_cmap = mpl.colors.ListedColormap(flat_cmap)
segmented_cmap = mpl.cm.get_cmap('Spectral_r',subdivision) # Colormap



# 1..) Gmap
import cartopy.io.img_tiles as cimgt
marker_size = 12
dataset_colors = {'nonna':'#14756D',
                 'multibeam_baie':'#659642',
                 'multibeam_river':'#27224B',
                 'hydroball':'#E86A2C',
                 'drone':'#F1B74B',
                 'lidar':'#e34234'}
                 
                 



# 1.a) Drone :
'#ffc16f'
axes[0].scatter(drone_da.Easting,
                drone_da.Northing,
                c = dataset_colors['drone'], s = marker_size, 
                label = 'Photogrammétrie par drone',
                transform = MTM7crs)


# 1.b) NONNA :
'#6495ed'
axes[0].scatter(nonna_da.Easting,
                nonna_da.Northing,
                c = dataset_colors['nonna'], s = marker_size, 
                label = 'Données bathymétriques NONNA',
                transform = MTM7crs)


# 1.c) Données LIDAR :

da_lidar = xr.concat([da_lidarc,da_lidarp], dim='ID')
axes[0].scatter(da_lidar.Easting,
                da_lidar.Northing,
                c = dataset_colors['lidar'], s = marker_size, 
                label = 'LiDAR bathymétrique',
                transform = MTM7crs)

## 1.f) Large multifaisceaux :
axes[0].scatter(large_da.Easting,
                large_da.Northing,
                c = '#04e7ad', s = marker_size,
                label = 'Échosondeur multifaisceaux (Large)',
                transform = MTM7crs)

# 1.g) Baie multifaisceaux :
'#04e7ad'
axes[0].scatter(baie_da.Easting,
                baie_da.Northing,
                c = dataset_colors['multibeam_baie'], s = marker_size, 
                label = 'Échosondeur multifaisceaux (Baie)',
                transform = MTM7crs)

# 1.e) Rivière multifaisceaux :
'orange'
axes[0].scatter(riviere_da.Easting,
                riviere_da.Northing,
                c = dataset_colors['multibeam_river'], s = marker_size,
                label = 'Échosondeur multifaisceaux (Rivière)',
                transform = MTM7crs)


# 1.d) Hydro :
'#ff7070'
axes[0].scatter(hydro_da.Easting,
                hydro_da.Northing,
                c = dataset_colors['hydroball'], s = marker_size, 
                label = 'Hydroball',
                transform = MTM7crs)

"""
# 2.0 :: Opening data to save time :
ds = xr.open_dataset('Topobathymetrie_finale/interpolation_highres.nc')
gridz = ds.topobathy.values
"""

# 2. Plotting grid interpolation : 
levels = np.array([-100] + list(np.linspace(-10,10,subdivision)) + [100])
im1 = axes[1].contourf(X,Y,gridz,
                       levels,
                       cmap=flat_cmap,
                       transform = MTM7crs,
                       vmin = -10, vmax=10,
                       zorder=1)
"""axes[1].contour(X,Y,gridz,subdivision,
                linewidths=0.3,colors='k',zorder=2,
                transform = MTM7crs)"""
#cb = fig.colorbar(im1, ax = axes[1], extend='both')
cb = fig.colorbar(cm.ScalarMappable(norm=im1.norm, cmap=segmented_cmap), ax=axes[1],extend='both')
cb.set_label('Niveau topobathymétrique (m)')



# 4. Fine tunning
for axe in axes :
    axe.set_extent(extent)
#request1 = cimgt.GoogleTiles(style='street',desired_tile_form='RGBA')
request1 = cimgt.GoogleTiles(style='satellite')
request2 = cimgt.GoogleTiles(style='satellite')
axes[0].add_image(request1, 15, alpha=0.60, zorder = 0)
axes[1].add_image(request2, 15, alpha=0.60, zorder = 0)

axes[0].set_title('Différents jeux de données')
axes[1].set_title('Interpolation finale')
lgd = axes[0].legend(loc='upper left', frameon = True, labelcolor='w', fancybox=False, prop=dict(weight='bold'), framealpha = None)
lgd.get_frame().set_alpha(None)
lgd.get_frame().set_edgecolor((1., 1., 1., 0.8))
lgd.get_frame().set_facecolor((0., 0., 0., 0.1))
#lgd = axes[0].legend(loc='upper left', frameon = True, fancybox=False)
fig.tight_layout()
plt.show()
### Le problème, c'est que le framealpha override tout le reste...
### La réponse est peut-être ici : https://stackoverflow.com/questions/58455158/legend-with-transparent-frame-and-non-transparent-edge

# 5. Sauvegarde de la figure :
savedir = '../6_Figures_rapport/2_Figures_topobathy/'
fig.savefig(savedir + 'interpolation_finale.png')


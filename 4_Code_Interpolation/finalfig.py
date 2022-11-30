# ==== IMPORTATION DES MODULES ====
import cmocean



# ==== CREATION DE LA FIGURE ====
print('CREATION FIGURE')
marker_size = 12
subdivision = 40
figsize = (11,6)
fig,axes = plt.subplots(1,2,figsize=figsize,
                        subplot_kw=dict(projection = Orthographic))

# Colormap limits :
c_cmap = 0.2
absval = c_cmap*max(abs(da.min()),abs(da.max()))
absval = 10


gl0 = axes[0].gridlines(draw_labels=True, linestyle = ':',color='k')
gl1 = axes[1].gridlines(draw_labels=True, linestyle = ':',color='k')
gl0.top_labels = gl0.right_labels = False
gl1.top_labels = gl1.right_labels = False
gl0.ylabel_style = {'rotation': 90}
gl1.ylabel_style = {'rotation': 90}

my_cmap = cmocean.cm.tarn_r


# Get the colormap colors
my_cmap = cmap(np.arange(cmap.N))

# Set alpha
my_cmap[:,-1] = np.concatenate( (np.ones(int(7*cmap.N/8)),
                                 np.linspace(1, 0.20, int(cmap.N/8))) )

# Create new colormap
my_cmap = mpl.colors.ListedColormap(my_cmap)

"""
color_list = my_cmap(np.linspace(0,1,subdivision))
cmap = my_cmap.from_list('new_cmap',color_list,subdivision)
cmap.set_over(color_list[-1][:3])
cmap.set_under(color_list[0][:3])
"""
#cmap = mpl.cm.get_cmap('Spectral_r',subdivision) # Colormap
#cmap = cmocean.cm.get_cmap('tarn_r',subdivision+2) # Colormap
extent = [-69.58, -69.545, 47.825, 47.86]


# 1..) Gmap
import cartopy.io.img_tiles as cimgt


# 1.a) Drone :
axes[0].scatter(drone_da.Easting,
                drone_da.Northing,
                c = '#ffc16f', s = marker_size,
                label = 'Photogrammétrie drone',
                transform = MTM7crs)

# 1.b) NONNA :
axes[0].scatter(nonna_da.Easting,
                nonna_da.Northing,
                c = '#6495ed', s = marker_size,
                label = 'NONNA',
                transform = MTM7crs)

# 1.c) Données LIDAR : 
da_lidar = xr.concat([da_lidarc,da_lidarp], dim='ID')
axes[0].scatter(da_lidar.Easting,
                da_lidar.Northing,
                c = '#e34234', s = marker_size,
                label = 'LIDAR découpées',
                transform = MTM7crs)

## 1.f) Large multifaisceaux :
#axes[0].scatter(large_da.Easting,
#                large_da.Northing,
#                c = '#04e7ad', s = marker_size,
#                label = 'Multifaisceaux (Large)',
#                transform = MTM7crs)

# 1.g) Baie multifaisceaux :
axes[0].scatter(baie_da.Easting,
                baie_da.Northing,
                c = '#04e7ad', s = marker_size,
                label = 'Capteur multifaisceaux (Baie)',
                transform = MTM7crs)
# 1.d) Hydro : 
axes[0].scatter(hydro_da.Easting,
                hydro_da.Northing,
                c = '#ff7070', s = marker_size,
                label = 'Hydroball',
                transform = MTM7crs)

# 1.e) Rivière multifaisceaux :
axes[0].scatter(riviere_da.Easting,
                riviere_da.Northing,
                c = 'orange', s = marker_size,
                label = 'Capteur multifaisceaux (Rivière)',
                transform = MTM7crs)




# 2. Plotting grid interpolation : 
levels = np.array([-100] + list(np.linspace(-absval,absval,subdivision)) + [100])
im1 = axes[1].contourf(X,Y,gridz,
                       levels,
                       cmap=my_cmap,
                       transform = MTM7crs,
                       vmin = -absval, vmax=absval,
                       zorder=1)
"""axes[1].contour(X,Y,gridz,subdivision,
                linewidths=0.3,colors='k',zorder=2,
                transform = MTM7crs)"""
#cb = fig.colorbar(im1, ax = axes[1], extend='both')
cb = fig.colorbar(cm.ScalarMappable(norm=im1.norm, cmap=im1.cmap), ax=axes[1],extend='both')


"""
# 3. Plot du brise-lame :
coords = np.array([[47.839775, -69.554181],
                   [47.842376, -69.553161],
                   [47.841487, -69.553380],
                   [47.841257, -69.551972]])
axes[1].plot(coords[:,1],coords[:,0],
             color = 'blue',
             label = 'Brise-lames',
             linewidth = 3,
             transform = PlateCarree,
             zorder=3)
"""
"""
# 4. Plot données manquantes (Marina) : 
coords = np.array([[47.847906, -69.567251],
                   [47.847935, -69.569976],
                   [47.846920, -69.572208],
                   [47.846193, -69.572090],
                   [47.846357, -69.567221]])
axes[1].fill(coords[:,1],coords[:,0],
             color = 'red', fill = False,
             label = 'Données manquantes (Marina R.-d.-L.)',
             linewidth = 3,
             transform = PlateCarree,
             zorder=3)
"""
                  
# 4. Fine tunning
cbar.ax.set_ylabel('Altitude [m]', rotation=270)
for axe in axes :
    axe.set_extent(extent)
request1 = cimgt.GoogleTiles(style='street',desired_tile_form='RGBA')
request2 = cimgt.GoogleTiles(style='satellite')
axes[0].add_image(request1, 16, alpha=0.5, zorder = 0)
axes[1].add_image(request2, 16, alpha=0.4, zorder = 0)

axes[0].set_title('Illustration des différents jeux de données')
axes[1].set_title('Interpolation finale post-débiaisage')
axes[0].legend(loc='upper left', framealpha = 0.7, labelcolor='k')
fig.tight_layout()
plt.show()

# 5. Sauvegarde de la figure :
savedir = '../6_Figures_rapport/2_Figures_topobathy/'
fig.savefig(savedir + 'interpolation_finale.png')

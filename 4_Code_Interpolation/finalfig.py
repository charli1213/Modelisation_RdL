
# ==== CREATION DE LA FIGURE ====
print('CREATION FIGURE')
marker_size = 12
subdivision = 30
figsize = (14,8)
fig,axes = plt.subplots(1,2,figsize=figsize,
                        subplot_kw=dict(projection = Orthographic))
gl0 = axes[0].gridlines(draw_labels=True, linestyle = ':',color='k')
gl1 = axes[1].gridlines(draw_labels=True, linestyle = ':',color='k')
gl0.top_labels = gl0.right_labels = False
gl1.top_labels = gl1.right_labels = False
gl0.ylabel_style = {'rotation': 90}
gl1.ylabel_style = {'rotation': 90}
#cmap = cm.get_cmap('RdYlBu_r',subdivision) # Colormap
cmap = cm.get_cmap('RdBu_r',subdivision) # Colormap
cmap = cm.get_cmap('seismic',subdivision) # Colormap
extent = [-69.585, -69.54, 47.82, 47.865]


# Colormap limits : 
absval = max(abs(da.min()),abs(da.max()))
# 1..) Gmap
import cartopy.io.img_tiles as cimgt
request1 = cimgt.GoogleTiles(style='street',desired_tile_form='RGBA')
request2 = cimgt.GoogleTiles(style='satellite')
axes[0].add_image(request1, 16, alpha=0.5)
axes[1].add_image(request2, 16, alpha=0.4)

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
                label = 'Hydrobole',
                transform = MTM7crs)

# 1.e) Rivière multifaisceaux :
axes[0].scatter(riviere_da.Easting,
                riviere_da.Northing,
                c = 'orange', s = marker_size,
                label = 'Capteur multifaisceaux (Rivière)',
                transform = MTM7crs)




# 2. Plotting grid interpolation : 
im1 = axes[1].contourf(X,Y,gridz,subdivision,
                       cmap=cmap,
                       transform = MTM7crs,
                       vmin = -absval, vmax=absval)
axes[1].contour(X,Y,gridz,subdivision,
                linewidths=0.3,colors='k',
                transform = MTM7crs)
cbar = fig.colorbar(cm.ScalarMappable(norm=im1.norm, cmap=im1.cmap), ax=axes[1],extend='both')




# Fine tunning
# Colorbar


cbar.ax.set_ylabel('Altitude [m]', rotation=270)
for axe in axes :
    axe.set_extent(extent)
axes[0].set_title('Illustration des données existantes')
axes[1].set_title('Données interpolées')
axes[0].legend(loc='upper left')
fig.tight_layout()
plt.show()




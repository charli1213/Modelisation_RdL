# ==== Figure embranchement ====
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
Orthographic = ccrs.Orthographic(-65,47) # Projection Orthographique
import cartopy.io.img_tiles as cimgt
request2 = cimgt.GoogleTiles(style='satellite')

""" À utiliser en faisant '%load embranchement_fig.py' après avoir fait rouler interp_bathy.py dans ipython """


marker_size = 20
fig,ax = plt.subplots(1,1,figsize=(7,9),
                        subplot_kw=dict(projection = Orthographic,
                                        axisbelow = True))
gl0 = ax.gridlines(draw_labels=True, linestyle = ':',color='k',zorder=0,linewidth=0.4)
gl0.top_labels = gl0.right_labels = False
#gl0.xlabel_style = {'rotation': 45}
gl0.ylabel_style = {'rotation': 90}

ax.set_extent([-69.585, -69.543, 47.82, 47.86], crs=ccrs.PlateCarree())
ax.add_image(request2, 16, alpha=0.45)
# 1.a) Drone :
ax.scatter(drone_da.Easting,
             drone_da.Northing,
             c = '#ffc16f', s = marker_size, 
             label = 'Données photogrammétrie par drone',
             transform = MTM7crs)
# 1.g) Baie multifaisceaux :
ax.scatter(baie_da.Easting,
             baie_da.Northing,
             c = '#04e7ad', s = marker_size, 
             label = 'Données capteur multifaisceaux',
             transform = MTM7crs)
# 1.d) Hydro : 
ax.scatter(hydro_da.Easting,
             hydro_da.Northing,
             c = '#ff7070', s = marker_size, 
             label = 'Données hydroball',
             transform = MTM7crs)
ax.set_title('Chevauchement des différents jeux de données')
ax.legend(loc='upper left', frameon=False, labelcolor='white')
ax.text(-0.07, 0.55, 'Latitude', va='bottom', ha='center',
        rotation='vertical', rotation_mode='anchor',
        transform=ax.transAxes)
ax.text(0.5, -0.07, 'Longitude', va='bottom', ha='center',
        rotation='horizontal', rotation_mode='anchor',
        transform=ax.transAxes)
fig.tight_layout()
plt.show()

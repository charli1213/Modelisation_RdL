#!/usr/bin/env python
# --- Importation des modules nécessaires :
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
from cartopy.io import shapereader
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

def make_map(projection=ccrs.PlateCarree()):
    fig, ax = plt.subplots(figsize=(9, 13),
                           subplot_kw=dict(projection=projection))
    gl = ax.gridlines(draw_labels=True, linestyle = ':')
    gl.xlabels_top = gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    return fig, ax
import cartopy.io.img_tiles as cimgt


extent = [-69.60,-69.54,47.82,47.89]

request = cimgt.GoogleTiles(style='satellite')

fig, ax = make_map(projection=request.crs)
ax.set_extent(extent)

ax.add_image(request, 13)

# --- Importation des modules
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

# Ça prend tous les fichiers
location = '/home/charles-edouard/Desktop/TraitementRDL/Trait_de_cote/'
filename = 'TC_20210628_20210708_WGS84.shp'
shapefile = gpd.read_file(location+filename)
#shapefile = gpd.read_file("20211004/Trait_côte_20211004.shx")
# Ça c'est un géodataframe

shapegeo = shapefile.geometry
# Ça c'est une GeoSeries qui contient 1 seul MultiLineString


a = [i for i in shapegeo[0][:]]

x = []
y = []
for lines in a :
    for tup in lines.coords : 
        x += [tup[0]]
        y += [tup[1]]

x = np.array(x)
y = np.array(y)
        
plt.plot(x,y)
plt.show()
#shapegeo.plot()
#plt.grid()
#plt.show()

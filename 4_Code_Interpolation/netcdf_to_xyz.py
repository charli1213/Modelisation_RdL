import numpy as np
import scipy
import xarray as xr
import matplotlib.pyplot as plt

path_str = 'Topobathymetrie_finale/'
filename = path_str + 'interpolation_highres.nc'
ds = xr.open_dataset(filename)

ds = ds.stack(ID=('xmtm7','ymtm7')).dropna('ID')



xpoints = ds.xmtm7.values
ypoints = ds.ymtm7.values
zpoints = ds.topobathy.values
midID = int(2*len(xpoints)/3)
topo_array1 = np.concatenate( [[xpoints[:midID:2]],
                               [ypoints[:midID:2]],
                               [zpoints[:midID:2]]],0).transpose()
topo_array2 = np.concatenate( [[xpoints[midID:2*midID:2]],
                               [ypoints[midID:2*midID:2]],
                               [zpoints[midID:2*midID:2]]],0).transpose()
topo_array3 = np.concatenate( [[xpoints[2*midID:]],
                               [ypoints[2*midID:]],
                               [zpoints[2*midID:]]],0).transpose()

# Saving divided topobathymetry xyz
np.savetxt(path_str + 'topobathy1.xyz',topo_array1,delimiter=' ')
np.savetxt(path_str + 'topobathy2.xyz',topo_array2,delimiter=' ')
np.savetxt(path_str + 'topobathy3.xyz',topo_array3,delimiter=' ')


topo_fusion = np.concatenate([topo_array1,topo_array2,topo_array3], 0)
np.savetxt(path_str + 'topobathy_limited.xyz',topo_fusion,delimiter=' ')

topo_fusion2 = -1*np.concatenate([topo_array1,topo_array2,topo_array3], 0)
np.savetxt(path_str + 'bathymetry_limited.xyz',topo_fusion2,delimiter=' ')

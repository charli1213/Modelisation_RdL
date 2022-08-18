### Important si le fichier topobathy_drone.nc a pas été créé ###
import rioxarray
import xarray as xr
from dask.diagnostics import ProgressBar

# Chemins : 
filepath = '../2_Donnees_entrantes/Topobathy_drone/'
filename = filepath + 'RDL_20220418_10CM.tif'
output_path  = '../5_Donnees_sortantes/'

# Opérations : 
da = xr.open_dataarray(filename, engine = 'rasterio', chunks='auto')
print('Fichier ouvert, transformation en netcdf.')

write_job = da.to_netcdf(output_path + 'topobathy_drone.nc', compute = False)
with ProgressBar():
    print("Écriture vers {}".format(output_path + 'topobathy_drone.nc'))
    write_job.compute()

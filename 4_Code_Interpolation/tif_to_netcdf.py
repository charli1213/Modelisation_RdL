# Ce code a pour but de transformer les fichier tiff nécessaire pour la création de la
# topobathymétrie de Rivière du Loup en NetCDF pour rendre la lecture avec Xarray
# beaucoup plus simple.

# Utile pour le tif de topobathy_drone et le lidar. Le résultat est 3 netCDF dans le
# dossier '5_données_sortantes'.

# --- Importation des modules
import rioxarray
import xarray as xr
from dask.diagnostics import ProgressBar

# --- Chemins fichiers entrant ::
path_drone = '../2_Donnees_entrantes/Topobathy_drone/'
file_drone = 'RDL_20220418_10CM_bonnnn.tif'
path_lidar = '../2_Donnees_entrantes/Lidar_Avion/'
file_lidar1 = 'Pointe_de_RdL_LIDAR_MTM7.tif'
file_lidar2 = 'cote_sud_LIDAR_MTM7.tif'

# --- Chemins fichiers sortant ::
output_path  = '../5_Donnees_sortantes/Netcdf/'
outfile_drone = 'topobathymetrie_drone.nc'
outfile_pointe = 'topographie_lidar_pointe_de_RdL.nc'
outfile_cote_s = 'topographie_lidar_cote_sud.nc'

# --- Opérations ::
da_drone = xr.open_dataarray(path_drone  + file_drone,  engine = 'rasterio', chunks='auto')
da_lidar1 = xr.open_dataarray(path_lidar + file_lidar1, engine = 'rasterio', chunks='auto')
da_lidar2 = xr.open_dataarray(path_lidar + file_lidar2, engine = 'rasterio', chunks='auto')
print('Ouverture des fichier :: {},{} et {}'.format(file_drone,
                                                    file_lidar1,
                                                    file_lidar2))

write_job_drone = da_drone.to_netcdf(output_path   + outfile_drone,  compute = False)
write_job_lidar1 = da_lidar1.to_netcdf(output_path + outfile_pointe, compute = False)
write_job_lidar2 = da_lidar2.to_netcdf(output_path + outfile_cote_s, compute = False)

with ProgressBar():
    print("Écriture vers {}".format(output_path + outfile_drone))
    write_job_drone.compute()

with ProgressBar():
    print("Écriture vers {}".format(output_path + outfile_pointe))
    write_job_lidar1.compute()

with ProgressBar():
    print("Écriture vers {}".format(output_path + outfile_cote_s))
    write_job_lidar2.compute()

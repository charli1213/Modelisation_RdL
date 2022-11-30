# === Importation des modules : 
import scipy.io
import numpy as np
from datetime import datetime,timedelta
import xarray as xr
import pandas as pd 
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


# === Definition fonctions : 
def numbers_to_datetime(number_array, padding = 367) :
    """ Transform number type array into datetime array
    padding (int) :: number of days to pad."""
    year0 = datetime.min 
    padding = timedelta(days = padding)
    dates = np.array([year0 + timedelta(days = float(nb))  for nb in number_array])
    dates = np.array([date - padding for date in dates])
    return  dates 

def datetime_to_numbers(dates_array) :
    """ Transforms datetimes into 'days form 2000-01-01 """
    date_init = np.datetime64('2000-01-01')
    #print(type(date_init))
    #print(type(dates_array[0]))
    number_array = np.array([(date-date_init)/np.timedelta64(1,'D') for date in dates_array])
    return number_array



# === Ouverture du fichier bouée spotter :
#     (Un seul fichier mat)panda_ds['Date/Time (LST)']
mat = scipy.io.loadmat('parametres_vagues_bouée_RDL.mat')

# --- Transformation en Xarray.Dataset :
print(mat.keys())
var_list = ['Hs', 'tetha_mean','tetha_peak']
time = numbers_to_datetime(mat['tm'].transpose())
print('Début : ',time[0],'\n','Fin  : ',time[-1])
lenght = len(time)
data_dict = {var_name:(['time'],mat[var_name].reshape(lenght)) for var_name in var_list}
waves_ds = xr.Dataset(data_dict, coords = {'time':time})



# === Ouverture des données de vent pour RdL :
#     (24 fichiers indépendants)
#     (Voir donnees_gouv.ssh pour plus d'info)
climate_data = []
for year in [2021,2022] :
    for month in range(1,13) :
        filename = 'en_climate_hourly_QC_7056616_{:2}-{}_P1H.csv'.format(str(month).zfill(2),year)
        climate_data += [pd.read_csv(filename,
                                     header=0,
                                     index_col=0,
                                     sep=",")]
climate_data = pd.concat(climate_data)


# --- Transformation en Xarray.Dataset
### Le temps est en LST, va falloir réparer tout ça.
time = np.array([datetime.strptime(datestr, '%Y-%m-%d %H:%M') for datestr in climate_data['Date/Time (LST)']])
time = time + timedelta(hours=5) # Correction UTC vs LTC
var_list = ['Temp (°C)','Wind Dir (10s deg)','Wind Spd (km/h)']
climate_data_dict = {var_name:(['time'],climate_data[var_name]) for var_name in var_list}
climate_ds = xr.Dataset(climate_data_dict, coords = {'time':time})



# === Ouverture des données de marée pour RdL :
tidefile = 'RDL_TIDE_1980_2022.mat'
tidemat = scipy.io.loadmat(tidefile)
tidetime = tidemat['TD'][:,0] # [sec]
tidelvl  = tidemat['TD'][:,1] # [ m ]
tidetime = numbers_to_datetime(tidetime)
tide_da = xr.DataArray(tidelvl, coords = [tidetime], dims = ['time'])
tide_da.name = 'Tide Level (m)'
#tide_ds = xr.Dataset({'tidelevel':(['time'], tidelvl)},
#                     coords = {'time':('time',tidetime)})


# === Exporting as CSV :
export_dir = 'CSV_Delft3D_Inputs/'
tide_da.to_dataframe().to_csv(export_dir + 'tide.csv')
climate_ds['Wind Spd (km/h)'].dropna('time').to_dataframe().to_csv(export_dir + 'wind.csv')


if __name__ == "__main__" :
    # --- Plot Wind speed
    f = interp1d(datetime_to_numbers(climate_ds.time.values), climate_ds['Wind Spd (km/h)'])
    new = f(datetime_to_numbers(waves_ds.time.values))
    plt.scatter(waves_ds['Hs'],new)
    plt.xlabel('Wave height (m)')
    plt.ylabel('Wind speed (m)')
    plt.grid(linestyle = ':')
    plt.show()
    plt.close()

    # --- Plot Wave Height
    fig, ax0 = plt.subplots()
    ax1 = ax0.twinx()
    waves_ds['Hs'].plot(label = 'Wave Height', ax = ax1, c='r')
    climate_ds['Wind Spd (km/h)'].plot( label = "Wind speed ['Km/h']", ax = ax0, c='b')
    plt.grid(linestyle = ':')
    plt.show()
    plt.close()
    
    # --- Plot tide
    tide_da.isel(time = slice(-1000,-1)).plot()
    plt.grid(linestyle = ':')
    plt.show()

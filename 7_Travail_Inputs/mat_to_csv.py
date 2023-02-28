# === Importation des modules : 
import scipy.io
import numpy as np
from datetime import datetime,timedelta
import xarray as xr
import pandas as pd 
import matplotlib.pyplot as plt
plt.rcParams['axes.axisbelow'] = True
from scipy.interpolate import interp1d
from random import random
#
from pyproj import Transformer
import cartopy._epsg as cepsg
import cartopy.crs as ccrs
#
MTM7crs  = cepsg._EPSGProjection(32187)   # Projection MTM7
PlateCarree = ccrs.PlateCarree()         # Projection Mercator
transformer_toUTM = Transformer.from_crs(MTM7crs,PlateCarree)

# Position du RBR au large
x,y = transformer_toUTM.transform(374634.174,5300539.484)
print('Position RBR_large ::',x,y)


# Position de la bouée :
x,y = transformer_toUTM.transform(374634.174,5300539.484)


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

def extend_nan(da, extension = 1) :
    """ 
    Prend un vecteur quelconque et 'diffuse' les nan à 
    l'intérieur de celui-ci. Par exemple : le vecteur
    [1, 2, 2, NaN, 5, 8, 9, 3, NaN] devient
    [1, 2, NaN, NaN, NaN, 8, 9, NaN, NaN]
    """
    # 1. On crée un masque :# Ici c'est juste des 1 et des nan.
    mask = da.notnull()
    mask = mask.where(mask == 1, np.nan)
    mask = mask.roll(time=1)*mask.roll(time=-1)*mask # On 'diffuse'
    masked_da = da*mask

    if extension == 1 : 
        return masked_da    
    else :
        return extend_nan(masked_da, extension=extension-1)
# === (Fin) Définition fonctions === #


    


# === 1. Ouverture du fichier bouée spotter (les vagues) :
#     (Un seul fichier mat)
position = pd.read_csv('Bouee_Spotter/rdl_location.csv',
                       index_col=False,)
#                       header=0,
#                       sep=",")
position['time'] = pd.to_datetime(pd.DataFrame({'year':position['# year'],
                                                'month':position['month'],
                                                'day':position['day'],
                                                'hour':position['hour'],
                                                'minute':position['min'],
                                                'second':position['sec'],
                                                'microsecond':position['msec']/1000,
                                                }))
posds = xr.Dataset(position.iloc[300:]).swap_dims({'dim_0':'time'}).drop('dim_0')



spottermat = scipy.io.loadmat('Bouee_Spotter/parametres_vagues_bouée_RDL.mat')

# --- Transformation en Xarray.Dataset :
print(spottermat.keys())
var_list = ['Hs','Tm01','Tm02', 'mds', 'pds_peak', 'tetha_mean', 'tetha_peak', 'tm', 'tp'] # Hs réfère à la 'Significant Wave Height'
time = numbers_to_datetime(spottermat['tm'].transpose())

data_dict = {var_name:(['time'],spottermat[var_name].reshape(len(time)), {'long_name': var_name + ' (Bouée Spotter V2, ELNAR)'}) for var_name in var_list}
spotterds = xr.Dataset( data_dict, coords = {'time':time}, attrs = {'name':r'Bouée Spotter V2 (Hs, Tm01,$\theta_p$) [ELNAR]'})



# === 2. Ouverture des données de vent pour RdL :
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
var_list = ['Temp (°C)','Wind Dir (10s deg)','Wind Spd (km/h)','Stn Press (kPa)','Temp (°C)']
climate_data_dict = {var_name : ( ['time'], climate_data[var_name], {'long_name':var_name + ' (ECCC-SMC)'}) for var_name in var_list}
climate_ds = xr.Dataset(climate_data_dict,
                        coords = {'time':time},
                        attrs={'name':r'Données Climat ($\vec{u}_{10}$, Temp, P) [ECCC-SMC]'}).dropna('time')
climate_ds['Wind Spd (m/s)'] = climate_ds['Wind Spd (km/h)']*1000/3600


# === 3. Ouverture des données de marée pour RdL :
### Les marées sont pas en CGVD28, elles sont en zéro des cartes (not de same)
tidefile = 'RDL_TIDE_1980_2022.mat'
tidemat = scipy.io.loadmat(tidefile)
tidetime = tidemat['TD'][:,0] # [sec]
tidelvl  = tidemat['TD'][:,1] # [ m ]
tidetime = numbers_to_datetime(tidetime)
tide_da = xr.DataArray(tidelvl,
                       coords = [tidetime],
                       dims = ['time'], attrs = {'long_name':'Prévisions de marées (SHC)'})
tide_da.name = 'Tide Level (m)'
tide_ds = tide_da.to_dataset()
tide_ds.attrs = {'name':'Prévisions Marées [SHC]'}
#tide_ds = xr.Dataset({'tidelevel':(['time'], tidelvl)},
#                     coords = {'time':('time',tidetime)})



# === 4. Ouverture des données DalCoast/Water_Level :
# Contient Dalcoast additionné aux prévisions de marées (CGVD28)
# "biais","npas",datetime"
WL_Dalcoast_file = 'DalCoast/CSV/RDL_Water_level_1996_2015.csv'
format = '%d-%b-%Y %H:%M:%S'
WLDalcoast_df  = pd.read_csv(WL_Dalcoast_file)
WLDalcoast_df['time'] = pd.to_datetime(WLDalcoast_df.time,format = format)
WLDalcoast_ds = xr.Dataset(WLDalcoast_df, attrs = {'name':"Niveau de l'eau, modèle Dalcoast [ISMER]"}).swap_dims({'dim_0':'time'}).drop('dim_0')
#WLDalcoast_ds.WL.attrs = dict(name = 'Water Level (Dalcoast) [ISMER]')


# Y'a un peu d'info dans le DalCoast.mat :
"""
dalfile = 'DalCoast/Faten/data_sta_61'
dalmat  = scipy.io.loadmat(dalfile, mat_dtype=True)
daltime = dalmat['TD'][:,0] # [sec]
dallvl  = dalmat['WL_CGVD28']
#dallvl  = dalmat['TD'][:,1] # [ m ]
daltime = numbers_to_datetime(daltime)
dal_da = xr.DataArray(dallvl, coords = [daltime], dims = ['time'])
"""



# === 5. Ouverture des données Water Level x paramètres vagues des RBR :
RBRfile = 'RBR/PARAMETRES_VAGUES_RBR_LARGE.mat'
RBRmat = scipy.io.loadmat(RBRfile) # C'est un dictionnaire!
RBRdict = {key : (['time'],
                  RBRmat[key][:,0],
                  {'long_name':key+' (RBR ELNAR)'}) for key in ['Hs_large','Niveau_eau_5min_CGVD28_large','Time_large']}
RBRds = xr.Dataset(RBRdict, attrs = {'name':r'Données RBR Large (WL,Hs,$\theta_p$) [ELNAR]'}).set_coords('Time_large')
RBRds['time'] = numbers_to_datetime(RBRmat['Time_large'][:,0])


# --- Correction/Interpolation des données des RBR
RBRda_trimmed   = extend_nan(RBRds.Niveau_eau_5min_CGVD28_large,5)
RBRda_corrected = RBRda_trimmed.interpolate_na(dim='time',method='cubic')
RBRds['Niveau_eau_5min_CGVD28_large'] = RBRda_corrected


# === 6. Ouverture des données de débit de la rivière :
debit_dir = 'Debit_Riviere_RdL/'
debit_df = pd.read_csv(debit_dir + 'debit_rdl.csv')
debit_df['time'] = pd.to_datetime(debit_df['Date'], format = '%Y/%m/%d')
debit_ds = xr.Dataset(debit_df,
                      attrs = {'name':r'Debit rivière $(m^3/s)$ [MELCCFP]'}).swap_dims({'dim_0':'time'}).drop('dim_0')




# --- Exporting as CSV :
export_dir = 'CSV_Delft3D_Inputs/'
tide_da.to_dataframe().to_csv(export_dir + 'tide.csv')
climate_ds['Wind Dir (10s deg)'].values = climate_ds['Wind Dir (10s deg)'].values*10
climate_ds.dropna('time').to_dataframe().to_csv(export_dir + 'climate_data.csv')
RBRds.drop(['Time_large','Hs_large']).dropna('time').to_dataframe().to_csv(export_dir + 'RBR_WL_data_corrected.csv')
spotterds.to_dataframe().to_csv(export_dir + 'spotter.csv')

if __name__ == "__main__" :
    # --- Metafigure :
    figsize = (10,10)
    fig, axes = plt.subplots(nrows = 4, ncols = 1, figsize = figsize)
    
    # --- Plot tide from RBR :
    RBRda_corrected.plot(label = 'Coupé et interpolé',c='r', ax = axes[0])
    tide_da.plot(ax=axes[0], label = "Marée en Niveau Moyen de l'eau")
    axes[0].legend()
    #RBRda_trimmed.plot(label = 'Non interpolé',c='blue', ax = axes[0])
    #axes[0].legend()
    
    # --- Plot Wave Height and wind :
    climate_ds['Wind Spd (km/h)'].plot( label = "Wind speed ['Km/h']", ax = axes[1], c='b')
    spotterds['Hs'].plot(label = 'Wave Height', ax = axes[2], c='r')

    # --- Plot pour comparer la bouée aux RBR (Wave height)
    #spotterds.Hs.plot(label='Spotter V2', ax = axes[3])
    RBRds.Hs_large.plot(ax = axes[3])
    
    # --- Fine tunning :
    tdelta = timedelta(days = 10)
    #maxdate = pd.to_datetime(spotterds.time.max().values).to_pydatetime() - tdelta
    #mindate = pd.to_datetime(RBRds.time.min().values).to_pydatetime() + tdelta

    # Période calme
    mindate = datetime(2021,10,9)
    maxdate = datetime(2021,10,16)

    # Période énarvée en hostie :
    mindate = datetime(2021,11,5)
    maxdate = datetime(2021,11,9)

    for ax,title in zip(axes,['Marée (RBR)','Vent','Wave Height (RBR)','Wave Height (Spotter)']) :
        ax.set_title(title)
        ax.set_xlim([mindate,maxdate])
        ax.grid(linestyle = ':')
    [(ax.set_xticklabels(''),ax.set_xlabel('')) for ax in axes[:-1]]
    fig.tight_layout()
    plt.show()

    

    
    # --- Plot Wind speed
    f = interp1d(datetime_to_numbers(climate_ds.time.values), climate_ds['Wind Spd (km/h)'])
    new = f(datetime_to_numbers(spotterds.time.values))
    plt.scatter(spotterds['Hs'],new)
    plt.xlabel('Wave height (m)')
    plt.ylabel('Wind speed (m)')
    plt.grid(linestyle = ':')
    plt.show()
    plt.close()

    

    
    # --- Plot Gant de l'inventaire des échantillons :
    fig, axes = plt.subplots(ncols = 4, figsize= (14,4))
    gs = axes[0].get_gridspec()
    for ax in axes[:3] : ax.remove()
    axleft = fig.add_subplot(gs[:3])
    axright = axes[3]
    inventaire = [spotterds, climate_ds, WLDalcoast_ds, tide_ds, RBRds, debit_ds]
    labels,i = [''],0
    for ds in inventaire:
        print(ds.name)
        color = [0,0.65,0.8-i/12]+ [1.]
        #color[1]=0.85
        axleft.hlines(i/5, ds.time[0], ds.time[-1], label = ds.name, color = color, linewidth=12)
        axright.hlines(i/5, ds.time[0], ds.time[-1], label = ds.name, color = color, linewidth=12)
        labels += [ds.name]
        i+=1
    axright.set_xlim(datetime(2020,9,1),datetime(2023,3,1))
    axleft.set_xlim(datetime(1979,1,1),datetime(2020,9,1))
    axleft.set_yticks(ticks = np.linspace(-0.2,i/5,i+2), labels = labels+[''])
    axright.set_yticks(ticks = np.linspace(-0.2,i/5,i+2), labels = '')
    for ax in [axleft, axright] :
        ax.grid(linestyle = ':')
        ax.set_axisbelow(True)
        ax.tick_params(axis='x', labelrotation=90)
    #axright.legend(loc= 'best', fancybox=False)
    axleft.set_title('Inventaire et intervalle des échantillons')
    fig.tight_layout()
    fig.savefig('../6_Figures_rapport/4_Inventaire_données/inventaire.png')
    plt.show()
    

    
    # --- Map de la bouées :
    plt.rc('axes', axisbelow=True)
    import cartopy.io.img_tiles as cimgt
    import cartopy.crs as ccrs
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    Orthographic = ccrs.Orthographic(-70,48) # Projection Orthographique

    ymean = posds[' latitude (decimal degrees)'].mean()
    xmean = posds['longitude (decimal degrees)'].mean()
    extent = [xmean-0.05, xmean+0.05, ymean-0.05, ymean+0.05]
    extent = [-69.62, -69.545, 47.82, 47.86]
    
    fig,axe = plt.subplots(1,1,figsize=figsize,
                           subplot_kw=dict(projection = Orthographic),
                           zorder = 4)
    gl0 = axe.gridlines(draw_labels=True, linestyle = ':',color='k')
    
    axe.plot(posds['longitude (decimal degrees)'],
             posds[' latitude (decimal degrees)'],
             transform = ccrs.PlateCarree(),
             color = 'orange',
             label = 'Déplacement bouée Spotter')
    axe.set_extent(extent)
    request1 = cimgt.GoogleTiles(style='satellite')
    axe.add_image(request1, 16, alpha=0.60, zorder = 0)
    axe.legend()
    plt.show()
    
    
    # --- Tide plot (Pour comparer le RBR et les données du SHC):
    #tide_mean = RBRds.Niveau_eau_5min_CGVD28_large
    #tide_ds['Tide Level (m)']
    



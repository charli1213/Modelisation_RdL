import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

figsize = (7,4)
fig, ax = plt.subplots(1,1,figsize = figsize)

nx = 8

yy = np.concatenate((np.random.rand(int(nx/2)),
                     np.random.rand(int(nx/2))+3))
xx = np.linspace(0,7,nx)
x = np.linspace(0,7,1000)
y_lin = interp1d(xx,yy,kind='linear')(x)
y_cub = interp1d(xx,yy,kind='cubic')(x)


ax.scatter(xx,yy,label = 'Jeu de données', marker = 'D', c = 'k')
ax.plot(x,y_lin, linestyle = ':', label = 'Interpolation linéaire',c='b')
ax.plot(x,y_cub, linestyle = '--',label = 'Interpolation cubique' ,c='r')
ax.legend(loc = 'best')
ax.set_title("Exemple illustrant l'apparition d'erreurs\n associées à l'interpolation cubique")
ax.set_xlabel('Position x (m)')
ax.set_ylabel('Topobathymétrie (m)')
plt.grid(linestyle=':')
fig.tight_layout()
fig.show()
savepath = '/home/charles-edouard/Desktop/Traitement_RdL/6_Figures_rapport/'
fig.savefig(savepath + 'gibbs.png')

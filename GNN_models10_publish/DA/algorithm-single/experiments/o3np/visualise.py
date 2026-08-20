import numpy as np
import matplotlib.pyplot as plt
import calendar, datetime
# import matplotlib
# matplotlib.rcParams.update({"font.size": 16})

import sys
sys.path.append('../../../..')
from general_ae_info import ae_props
AE_props = ae_props('v63')

grid_lats = np.load("../../../../../grid_lats.npy")
grid_lons = np.load("../../../../../grid_lons.npy")
zerolon = np.nonzero(grid_lons == 0.)
glzl = grid_lats[zerolon]

# print(grid_lats[zerolon])

# print(grid_lats[zerolon][1:] - grid_lats[zerolon][:-1])

datadir = '../data/o3np'
general_name = 'ana_inc_and_bg_at_0W_'
general_name2 = '_60.0N'

year = 2023
winterm = 12
summerm = 6

winterdays = [f"{year:04d}-{winterm:02d}-{day:02d}-00" for day in range(1,  calendar.monthrange(year, winterm)[1] + 1)]
summerdays = [f"{year:04d}-{summerm:02d}-{day:02d}-00" for day in range(1,  calendar.monthrange(year, summerm)[1] + 1)]

print(winterdays)
print(summerdays)

def mean_ana_inc(days):
    len = 0
    for d in days:
        new_data = np.load(f'{datadir}/{general_name}{d}{general_name2}.npz')['ana_inc']
        if len == 0:
            ana_inc = new_data
        else:
            ana_inc = ana_inc * len / (len + 1) + new_data / (len + 1)
        len += 1
    
    return ana_inc

print(np.shape(mean_ana_inc(winterdays)))
print(np.shape(mean_ana_inc(summerdays)))

winterinc = mean_ana_inc(winterdays)
summerinc = mean_ana_inc(summerdays)

# From general_ae_info:
# self.levels = [5,10,20,30,50,70,100,150,200,300,400,500,600,700,800,850,925,1000] # pressure levels in hPa
# self.variables = ["u", "v", "z", "t", "q","o3"] # zonal wind, meridional wind, geopotential, temperature, specific humidity
# combinations = [f"{var}{level}" for var in self.variables for level in self.levels]
# # Surface variables
# surface_variables = ["t2m", "msl", "sd", "siconc2","sst2","stl1"]
# self.time_varying_variables = combinations + surface_variables
# self.reconstructed_variables = combinations + surface_variables

# Plot t and o3

levels = AE_props.levels
wt2d = np.zeros((winterinc.shape[0], len(levels)))
st2d = np.zeros((winterinc.shape[0], len(levels)))
wo2d = np.zeros((winterinc.shape[0], len(levels)))
so2d = np.zeros((winterinc.shape[0], len(levels)))
print(np.all(st2d))
print(len(np.where(wt2d == 0)))
ct = 0
for i in range(len(winterinc[1])):
    c = AE_props.reconstructed_variables[i]
    if c[0] == 't' and c != 't2m':
        wt2d[:,levels.index(int(c[1:]))] = winterinc[:,i]
        st2d[:,levels.index(int(c[1:]))] = summerinc[:,i]
    elif c[:2] == 'o3':
        wo2d[:,levels.index(int(c[2:]))] = winterinc[:,i]
        so2d[:,levels.index(int(c[2:]))] = summerinc[:,i]

# minmaxT = max(np.abs(wt2d), np.abs(st2d))
# minmaxo3 = max(np.abs(wo2d), np.abs(so2d))
# print(np.max(np.abs(wt2d)))
print(np.max(np.abs(wt2d)), np.max(np.abs(st2d)), np.max(np.abs(wo2d)), np.max(np.abs(so2d)))

levt = np.arange(-0.55, 0.551, step=0.1)
levo = np.arange(-1.5, 1.51, step=2e-1)
# plt.figure(figsize=(12,10))
plt.subplot(2,2,1)
plt.contourf(glzl, levels, st2d.T, cmap='bwr', levels=levt, extend='both')
plt.xlabel('Latitude [°]')
plt.ylabel('Level [hPa]')
plt.colorbar(ticks=[0.45, 0.15, -0.15, -0.45], label='[K]')
plt.title('T increment, June')
plt.yscale('log')
plt.ylim(max(levels), min(levels))
plt.xlim(35, 85)
plt.scatter(60, 50, c='gold', marker='*', edgecolors='k', linewidths=0.4, zorder=1000)

plt.subplot(2,2,2)
plt.contourf(glzl, levels, wt2d.T, cmap='bwr', levels=levt, extend='both')
plt.xlabel('Latitude [°]')
plt.ylabel('Level [hPa]')
plt.colorbar(ticks=[0.45, 0.15, -0.15, -0.45], label='[K]')
plt.title('T increment, December')
plt.yscale('log')
plt.ylim(max(levels), min(levels))
plt.xlim(35, 85)
plt.scatter(60, 50, c='gold', marker='*', edgecolors='k', linewidths=0.4, zorder=1000)


plt.subplot(2,2,3)
plt.contourf(glzl, levels, so2d.T * 1e7, cmap='bwr', levels=levo, extend='both')
plt.xlabel('Latitude [°]')
plt.ylabel('Level [hPa]')
plt.colorbar(ticks=np.array([1.5, 0.9, 0.3, -0.3, -0.9, -1.5]), label=r'[$10^{-7}$ kg/kg]')
plt.title(r'$\mathrm{O}_{3}$ increment, June')
plt.yscale('log')
plt.ylim(max(levels), min(levels))
plt.xlim(35, 85)
plt.scatter(60, 50, c='gold', marker='*', edgecolors='k', linewidths=0.4, zorder=1000)

plt.subplot(2,2,4)
plt.contourf(glzl, levels, wo2d.T * 1e7, cmap='bwr', levels=levo, extend='both')
plt.xlabel('Latitude [°]')
plt.ylabel('Level [hPa]')
plt.colorbar(ticks=np.array([1.5, 0.9, 0.3, -0.3, -0.9, -1.5]), label=r'[$10^{-7}$ kg/kg]')
plt.title(r'$\mathrm{O}_{3}$ increment, December')
plt.yscale('log')
plt.ylim(max(levels), min(levels))
plt.xlim(35, 85)
plt.scatter(60, 50, c='gold', marker='*', edgecolors='k', linewidths=0.4, zorder=1000)

plt.tight_layout()

plt.savefig('mean_inc.pdf')
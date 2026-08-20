import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({"font.size": 16})

import sys
sys.path.append('../../../..')
from general_ae_info import ae_props
AE_props = ae_props('v63')

grid_lats = np.load("../../../../../grid_lats.npy")
grid_lons = np.load("../../../../../grid_lons.npy")

obs = 'u700'
reflevel = 700
dt = '2023-04-01-00'

data_dir = '../data/ekman/'
general_name = f'obs_{obs}_ana_inc_and_bg_at_'



# print((10 - 350 + 180) % 360 - 180)
# print((350 - 10 + 180) % 360 - 180)
# raise AssertionError

levels = AE_props.levels

def winddir(u,v):
    wind_dir = (270 - np.degrees(np.arctan2(v, u))) % 360
    return wind_dir


def dirchange(u, v, reflevel=reflevel, botlevel=1000):
    def wrapped_dif(lower, upper):
        return (lower - upper + 180) % 360 - 180

    wd = winddir(u,v)
    dtheta = wd[levels.index(botlevel)] - wd[levels.index(reflevel)]
    lower_winddir = wd[levels.index(reflevel):levels.index(botlevel)+1]
    if dtheta < 0:
        fracdtheta = sum(-np.array([min(i,0) for i in wrapped_dif(lower_winddir[1:], lower_winddir[:-1])])) / sum(np.abs(wrapped_dif(lower_winddir[1:], lower_winddir[:-1])))
    else:
        fracdtheta = sum(np.array([max(i,0) for i in wrapped_dif(lower_winddir[1:], lower_winddir[:-1])])) / sum(np.abs(wrapped_dif(lower_winddir[1:], lower_winddir[:-1])))
    return dtheta, fracdtheta

def load_one_instance(dt):
    lats = []
    lons = []

    u, v, speed = [], [], []
    dtheta, dtheta2 = [], []
    fracdtheta = []
    all_theta = []

    with open('filtered_coordinates.txt', 'r') as d:
        for line in d:
            lat, lon, idx = line.split()
            lat, lon = float(lat), float(lon)
            ana_inc = np.load(f'{data_dir}{general_name}{lat}_{lon}_{idx}_{dt}.npz')['ana_inc']
            # print(ana_inc.shape)
            

            u1 = np.zeros(len(levels))
            v1 = np.zeros(len(levels))

            for i in range(len(ana_inc)):
                c = AE_props.reconstructed_variables[i]
                #print(c)
                if c[0] == 'u':
                    # print(ana_inc[i])
                    # print(c, levels.index(int(c[1:])))
                    u1[levels.index(int(c[1:]))] = ana_inc[i]
                elif c[0] == 'v':
                    v1[levels.index(int(c[1:]))] = ana_inc[i]
            
            # print(levels)
            # print(u1[levels.index(700):])
            # print(v1[levels.index(700):])
            speed1 = np.sqrt(u1**2 + v1**2)
            # print(speed1[levels.index(700):])

            # print(winddir(u1,v1)[levels.index(700):])
            # print(dirchange(u1,v1))
            all_theta1 = winddir(u1,v1)#[levels.index(reflevel):]
            d1 = dirchange(u1,v1)
            d2 = dirchange(u1,v1,botlevel=925)

            u.append(u1)
            v.append(v1)
            speed.append(speed1)
            all_theta.append(all_theta1)
            dtheta.append(d1[0])
            dtheta2.append(d2[0])
            fracdtheta.append(d1[1])
            lats.append(lat)
            lons.append(lon)
    
    return lats,lons,u, v, speed,dtheta, dtheta2,fracdtheta,all_theta

lats,lons,u, v, speed,dtheta, dtheta2,fracdtheta,all_theta = load_one_instance(dt)
        
print(np.shape(u), np.shape(v), np.shape(speed), np.shape(all_theta), np.shape(dtheta), np.shape(fracdtheta))
# grid_lons = np.load("../../../../../grid_lons.npy")
# zerolon = np.nonzero(grid_lons == 0.)
# print(zerolon)
# fracdthetaarr = np.array([f for f in fracdtheta])
# print(np.nonzero(fracdthetaarr <= 1))
# print(lons[np.nonzero(fracdthetaarr == 1.)[0]])
fracdtheta = np.array(fracdtheta)
lons = np.array(lons)
lats = np.array(lats)
dtheta = np.array(dtheta)

print(np.nonzero(fracdtheta < 0.5))
print(np.amin(fracdtheta))

#print(np.nonzero(fracdtheta <= 1 and fracdtheta > 0.9))
import cartopy.crs as ccrs
proj = ccrs.Robinson()
#ax = plt.subplot
plt.figure(figsize=(12,6))
ax = plt.axes(projection=proj)

print(np.amin(dtheta), np.amax(dtheta))

criteria = [1, 0.95, 0.8, 0]
shapes = ['D', 'v', 'o', 'X']
sizes=[60, 25, 10]
labels=[r'$H>0.95$',r'$H>0.8$',r'$H\leq 0.8$']
for i in range(len(criteria)-1):
    s = ax.scatter(lons[np.nonzero((fracdtheta <= criteria[i]) & (fracdtheta > criteria[i+1]))],
                lats[np.nonzero((fracdtheta <= criteria[i]) & (fracdtheta > criteria[i+1]))], 
                c=dtheta[np.nonzero((fracdtheta <= criteria[i]) & (fracdtheta > criteria[i+1]))], 
                marker='o', cmap='seismic', s=sizes[i], vmin=-70, vmax=70, transform=ccrs.PlateCarree(),
                edgecolor='k', linewidth=0.3)#, label=labels[i])
    ax.scatter(1e3, 1e3, c='white', edgecolor='k', s=sizes[i], linewidth=0.3, label=labels[i], transform=ccrs.PlateCarree())
    if i == 0:
        plt.colorbar(s, label=r'$\vartheta_{1000} - \vartheta_{700} \,[\degree]$', fraction=0.02, pad=0.04)
# s = ax.scatter(lons, lats, c=dtheta, cmap='seismic', s=fracdtheta*35, vmin=-70, vmax=70, transform=ccrs.PlateCarree(),
#                edgecolor='k', linewidth=0.3)
# plt.colorbar(s)
ax.coastlines()
ax.plot([-180, 180], [-30, -30], color='gray', linewidth=3, linestyle='-', transform=ccrs.PlateCarree())
ax.plot([-180, 180], [30, 30], color='gray', linewidth=3, linestyle='-', transform=ccrs.PlateCarree())
ax.plot([-180, 180], [0, 0], color='gray', linewidth=3, linestyle='--', transform=ccrs.PlateCarree())
plt.legend()
plt.title('Wind veering in analysis increments')
plt.tight_layout()
plt.savefig('dtheta.pdf')# and fracdtheta > criteria[i+1])


nlatsidx = np.nonzero(lats > 30)
slatsidx = np.nonzero(lats < -30)
ntlatsidx = np.nonzero((lats < 30) & (lats > 0))
stlatsidx = np.nonzero((lats > -30) & (lats < 0))


ndtheta = dtheta[nlatsidx]
sdtheta = dtheta[slatsidx]

ntdtheta = dtheta[ntlatsidx]
stdtheta = dtheta[stlatsidx]

nfrac = fracdtheta[nlatsidx]
sfrac = fracdtheta[slatsidx]

ntfrac = fracdtheta[ntlatsidx]
stfrac = fracdtheta[stlatsidx]

mv = 10

print('NH')
for crit in criteria[1:]:
    print(crit)
    print(len(np.nonzero(nfrac > crit)[0]))
    ndt = ndtheta[np.nonzero(nfrac > crit)]
    print(len(np.nonzero(ndt > mv)[0]), len(np.nonzero(ndt < -mv)[0]))
    print(np.mean(ndt))
    print()


print('SH')
for crit in criteria[1:]:
    print(crit)
    print(len(np.nonzero(sfrac > crit)[0]))
    sdt = sdtheta[np.nonzero(sfrac > crit)]
    print(len(np.nonzero(sdt > mv)[0]), len(np.nonzero(sdt < -mv)[0]))
    print(np.mean(sdt))
    print()


# print('NH t')
# for crit in criteria[1:]:
#     print(crit)
#     print(len(np.nonzero(ntfrac > crit)[0]))
#     ntdt = ntdtheta[np.nonzero(ntfrac > crit)]
#     print(len(np.nonzero(ntdt > mv)[0]), len(np.nonzero(ntdt < -mv)[0]))
#     print(np.mean(ntdt))
#     print()


# print('SH t')
# for crit in criteria[1:]:
#     print(crit)
#     print(len(np.nonzero(stfrac > crit)[0]))
#     stdt = stdtheta[np.nonzero(stfrac > crit)]
#     print(len(np.nonzero(stdt > mv)[0]), len(np.nonzero(stdt < -mv)[0]))
#     print(np.mean(stdt))
#     print()



# AVERAGED STATISTICS
all_dt = [f'2023-{i:02d}-01-00' for i in range(1,13)]
mean_dtheta = np.zeros(dtheta.shape)
mean_dtheta2 = np.zeros(dtheta.shape)
l = 0
for dt in all_dt:
    new_dtheta = np.array(load_one_instance(dt)[5])
    mean_dtheta = (l * mean_dtheta + 1 * new_dtheta) / (l+1)
    new_dtheta2 = np.array(load_one_instance(dt)[6])
    mean_dtheta2 = (l * mean_dtheta2 + 1 * new_dtheta2) / (l+1)
    l += 1



plt.figure(figsize=(6,6))
ax = plt.axes(projection=proj)

# print(np.amin(dtheta), np.amax(dtheta))

# criteria = [1, 0.95, 0.8, 0]
# shapes = ['D', 'v', 'o', 'X']
# sizes=[60, 25, 10]
# labels=[r'$H>0.95$',r'$H>0.8$',r'$H\leq 0.8$']
# for i in range(len(criteria)-1):

cr = 5
subset1idx = np.nonzero(mean_dtheta - mean_dtheta2 > cr)
subset2idx = np.nonzero(mean_dtheta - mean_dtheta2 < -cr)
subset3idx = np.nonzero((mean_dtheta - mean_dtheta2 <= cr) & (mean_dtheta - mean_dtheta2 > -cr))
import matplotlib.ticker as mticker
s = ax.scatter(lons,
            lats, 
            c=mean_dtheta, 
            marker='o', cmap='seismic', s=10, vmin=-20, vmax=20, transform=ccrs.PlateCarree(),
            )#edgecolor='k', linewidth=1, linestyle='-')#, label=labels[i])
    # ax.scatter(1e3, 1e3, c='white', edgecolor='k', s=sizes[i], linewidth=0.3, label=labels[i], transform=ccrs.PlateCarree())
    # if i == 0:
plt.colorbar(s, label=r'$\vartheta_{1000} - \vartheta_{700} \,[\degree]$', fraction=0.02, pad=0.04, extend='both',
            ticks=[-20, -10, 0, 10, 20],)
            #format=mticker.FixedFormatter(['-20°', '-10°', '0°', '10°', '20°'])
#)
# plt.colorbar(s, label=r'$\vartheta_{1000} - \vartheta_{700} \,[\degree]$', shrink=0.7, location='bottom', pad=0.03, fraction=0.5, extend='both',#)
#              ticks=[-20, -10, 0, 10, 20],#)
#             format=mticker.FixedFormatter(['-20°', '-10°', '0°', '10°', '20°']),
#              )
# ax.scatter(lons[subset2idx],
#             lats[subset2idx], 
#             c=mean_dtheta[subset2idx], 
#             marker='o', cmap='seismic', s=35, vmin=-20, vmax=20, transform=ccrs.PlateCarree(),
#             edgecolor='gold', linewidth=1, linestyle='-')
# ax.scatter(lons[subset3idx],
#             lats[subset3idx], 
#             c=mean_dtheta[subset3idx], 
#             marker='o', cmap='seismic', s=35, vmin=-20, vmax=20, transform=ccrs.PlateCarree(),
#             edgecolor='k', linewidth=1, linestyle=':')
# s = ax.scatter(lons, lats, c=dtheta, cmap='seismic', s=fracdtheta*35, vmin=-70, vmax=70, transform=ccrs.PlateCarree(),
#                edgecolor='k', linewidth=0.3)
# plt.colorbar(s)
ax.coastlines()
# ax.plot([-180, 180], [-30, -30], color='gray', linewidth=3, linestyle='-', transform=ccrs.PlateCarree())
# ax.plot([-180, 180], [30, 30], color='gray', linewidth=3, linestyle='-', transform=ccrs.PlateCarree())
# ax.plot([-180, 180], [0, 0], color='gray', linewidth=3, linestyle='--', transform=ccrs.PlateCarree())
#plt.legend()
plt.title('Wind veering below 700 hPa')
plt.tight_layout()
plt.savefig('dtheta_mean.pdf')# and fracdtheta > criteria[i+1])

plt.cla()
plt.clf()
plt.figure(figsize=(6,6))
ax = plt.axes(projection=proj)
s = ax.scatter(lons,
            lats, 
            c=mean_dtheta - mean_dtheta2, 
            marker='o', cmap='seismic', s=10, vmin=-10, vmax=10, transform=ccrs.PlateCarree(),
            )#edgecolor='k', linewidth=1, linestyle='-')#, label=labels[i])
    # ax.scatter(1e3, 1e3, c='white', edgecolor='k', s=sizes[i], linewidth=0.3, label=labels[i], transform=ccrs.PlateCarree())
    # if i == 0:
plt.colorbar(s, label=r'$\vartheta_{1000} - \vartheta_{925} \,[\degree]$', fraction=0.02, pad=0.04, extend='both', ticks=[-10, -5, 0, 5, 10])
#plt.colorbar(s, label=r'$\vartheta_{1000} - \vartheta_{925} \,[\degree]$', shrink=0.45, location='bottom', pad=0.025, extend='both')
ax.coastlines()
plt.title('Wind veering in the lowest layer')
plt.tight_layout()
plt.savefig('dtheta_mean2.pdf')# and fracdtheta > criteria[i+1])

ndtheta = mean_dtheta[nlatsidx]
sdtheta = mean_dtheta[slatsidx]
ndtheta2 = mean_dtheta2[nlatsidx]
sdtheta2 = mean_dtheta2[slatsidx]

print(np.amin(mean_dtheta), np.amax(mean_dtheta))

print('NH')
print(len(np.nonzero(ndtheta > mv)[0]), len(np.nonzero(ndtheta < -mv)[0]))
print(len(ndtheta))
print(len(np.nonzero((ndtheta < -mv) & (ndtheta < 2*ndtheta2))[0]))
print(len(np.nonzero(ndtheta - ndtheta2 > cr)[0]), len(np.nonzero(ndtheta - ndtheta2 < -cr)[0]))
print()
print(len(np.nonzero(sdtheta > mv)[0]), len(np.nonzero(sdtheta < -mv)[0]))
print(len(sdtheta))
print(len(np.nonzero((sdtheta > mv) & (sdtheta > 2*sdtheta2))[0]))
print(len(np.nonzero(sdtheta - sdtheta2 > cr)[0]), len(np.nonzero(sdtheta - sdtheta2 < -cr)[0]))
print()

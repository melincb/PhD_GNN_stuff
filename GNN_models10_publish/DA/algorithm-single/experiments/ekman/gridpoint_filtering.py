import numpy as np
import sys
sys.path.append('../..')
from interpolator import normalize_longitudes


lsm = np.load('') # insert path to mask

lt = np.load('../../../../../grid_lats.npy')
ln = np.load('../../../../../grid_lons.npy')

print(ln)

latlist = lt[np.nonzero(ln == 0.)]

desired_lats = np.append(latlist[4:len(latlist)//2:4], latlist[len(latlist)//2+3::4])

ltf, lnf  = [], []
idx = []

print(np.any(np.abs(desired_lats - 86.9) < 0.1))
print(np.any(np.abs(desired_lats - 85.9) < 0.1))

last_lt = 0

for i in range(len(lt)):
    if abs(lt[i]) > 30:
        if np.any(np.abs(desired_lats - lt[i]) < 0.1):
            if len(ltf) == 0 or abs(lt[i] - last_lt) > 1e-3:
                # new latitude
                latcount = 0
            last_lt = lt[i]
            if latcount % 4 == 0 and lsm[i] == 0.:
                ltf.append(lt[i])
                lnf.append(ln[i])
                idx.append(i)
            latcount += 1

print(len(ltf))

filtered_coordinates = [(ltf[i], normalize_longitudes(lnf[i]), idx[i]) for i in range(len(ltf))]

with open("filtered_coordinates.txt", "w") as f:
    for x, y, i in filtered_coordinates:
        f.write(f"{x} {y} {i}\n")
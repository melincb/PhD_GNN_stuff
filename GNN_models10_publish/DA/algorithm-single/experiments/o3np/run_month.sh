#!/usr/bin/bash

echo "You should run me in experiments dir using bash experiments/o3np/run_month.sh" 

YEAR=2023
MONTH=06

# Get last day of the month
DAYS=$(date -d "$YEAR-$MONTH-01 +1 month -1 day" +%d)

for DAY in $(seq -w 1 "$DAYS"); do
    echo "${YEAR}-${MONTH}-${DAY}-00"
    python 3D_var_experiment.py --obs_datetime=${YEAR}-${MONTH}-${DAY}-00 --AE_version=v63 --custom_addon=o3np --obs_qty=o350 --obs_inc=0.000001 --obs_std=0.0000002 --singobs_lat=60.0 --singobs_lon=0.0 --B_type=climatological --init_lr=0.5
done


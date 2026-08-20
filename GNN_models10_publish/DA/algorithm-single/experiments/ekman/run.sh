#!/usr/bin/bash

echo "You should run me in experiments dir using bash experiments/ekman/run.sh" 

filename="experiments/ekman/filtered_coordinates.txt"

count=0
start=$(date +%s)

YEAR=2023

for MONTH in $(seq -w 1 12); do
    while read -r lat lon idx
    do
        echo "${YEAR}-${MONTH}-01-00 --singobs_lat=$lat --singobs_lon=$lon"
        current=$(date +%s)
        echo -e "\n iteration $count time elpsed $(($current-$start))"
        python 3D_var_experiment.py --obs_datetime=${YEAR}-${MONTH}-01-00 --AE_version=v63 --custom_addon=ekman$idx --obs_qty=u700 --obs_inc=5.0 --obs_std=1.0 --singobs_lat=$lat --singobs_lon=$lon --B_type=climatological --init_lr=0.5
        count=$((count+1))
    done < "$filename"
done

current=$(date +%s)
echo "\n total time elpsed $(($current-$start))"

#for DAY in $(seq -w 1 "$DAYS"); do
#    echo "${YEAR}-${MONTH}-${DAY}-00"
#    python 3D_var_experiment.py --obs_datetime=${YEAR}-${MONTH}-${DAY}-00 --AE_version=v63 --custom_addon=o3np --obs_qty=o350 --obs_inc=0.000001 --obs_std=0.0000002 --singobs_lat=60.0 --singobs_lon=0.0 --B_type=climatological --init_lr=0.5
#done


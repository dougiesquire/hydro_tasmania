#!/bin/bash

f3dir=/OSM/CBR/OA_DCFP/data2/model_output/CAFE/forecasts/f3/
newdir=/OSM/CBR/OA_DCFP/work/squ027/f3_precip

for read_path in $f3dir/yr2002/mn10/OUTPUT.10; do
    write_path=$newdir${read_path#$f3dir}
    mkdir -p $write_path

    echo "Extracting precip from $read_path..."
    for read_file in $read_path/atmos_daily*.nc; do
        write_file=$write_path${read_file#$read_path}
        ncks -v precip -O $read_file $write_file
    done
    
    echo "    concatenating in time..."
    concat_files=`ls $write_path/atmos_daily*.nc`
    concat_arr=($concat_files)
    ncrcat -O $concat_files ${concat_arr[0]%.nc}.precip.nc
    rm $concat_files
done


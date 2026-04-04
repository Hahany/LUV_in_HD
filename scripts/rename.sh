#!/bin/bash
root="/home/xiaochu/Public/project-LUV/data/output/0805/no_overlap_simu_diff/rh_0.8_rs_0.6_Lst_0_Len2.00_nseg_1000"
for dir in /home/xiaochu/Public/project-LUV/data/output/0805/no_overlap_simu_diff/rh_0.8_rs_0.6_Lst_0_Len2.00_nseg_1000/Rc_*; do
  dir1=${dir%/}
  dir1=${dir1##*/}
  echo $dir1
  mv ${dir}/entire_density.txt ${dir}/ld_str.txt
#  mv ${dir}/local_density.txt ${dir}/ld_all.txt
#  dir2=$(echo "$dir1" | sed 's/Rc_vs_deltaV_//')
#  echo "$root"/"$dir2"
#  mv "$dir" "$root"/"$dir2"

done

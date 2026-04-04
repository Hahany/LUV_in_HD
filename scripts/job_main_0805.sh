#!/bin/bash
#=============Set arguments=============================
export N_frame=100000
export N_seg=1000
export Rho=0.805
export MEM_LIMIT=$(( 400*1024*1024 ))
echo $MEM_LIMIT
export N_THREADS=140
#=============Step 1: INITIALIZE========================
#python3.11 ../apps_no_overlap/main_simu_diff.py 1 "$N_frame" "$N_seg" "$Rho"  # for main.py
#=============Step 2: Parallel==========================
run_main(){
  Rc=$1
  ulimit -v "$MEM_LIMIT"
  nice -n 19 python3.11 ../apps_no_overlap/main_simu_diff.py "$Rc" "$N_frame" "$N_seg" "$Rho"
}
export -f run_main
seq 17 140 | awk '{printf "%.2f\n", $1*0.5+1.00}' | \
nice -n 19 parallel --line-buffer -j "$N_THREADS" run_main

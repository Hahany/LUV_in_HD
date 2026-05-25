#!/bin/bash
#=============Step 1: INITIALIZE========================
#python3.11 ../apps_no_overlap/main_simu_diff.py 1 "$N_frame" "$N_seg" "$Rho"  # for main.py
#=============Step 2: Parallel==========================
run_main(){
  N_frame=$1
  N_seg=$2
  Rho=$3
  MEM_LIMIT=$(( 120*1024*1024 ))
  N_THREADS=$4
  Rc=$5
  ulimit -v "$MEM_LIMIT"
  nice -n 19 python3.11 /home/xiaochu/Public/LUV_in_HD/apps/main_measurement/main.py "$Rc" "$N_frame" "$N_seg" "$Rho"
}
export -f run_main
export -f run_main


MAX_JOBS=40
count=0
Nf_list=(100 500 10000)  # 原来的总帧数0.780~0.805 依次是 [1000, 1000, 10000, 10000, 100000, 100000]  当前选短一点，是应为如果全部纳入计算耗时过长，则只能选择短一些的帧数
for j in $(seq 0 0); do
    Rho=$(echo "$j * 0.005 + 0.780" | bc -l)
    for i in $(seq 0 60); do
        case $j in
          0|1)
            N_f=${Nf_list[0]}
            N_seg=100
            ;;
          2|3)
            N_f=${Nf_list[1]}
            N_seg=100
            ;;
          *)
            N_f=${Nf_list[2]}
            N_seg=1000
            ;;
        esac
        echo $N_f
        N_THREADS=40
        Rc=$(echo "$i * 0.5 + 1" | bc -l)
        run_main "$N_f" "$N_seg" "$Rho" "$N_THREADS" "$Rc" &

        ((count++))
        if (( count % MAX_JOBS == 0 )); then
            wait  # 每启动 MAX_JOBS 个任务，就等它们全部完成
        fi
    done
    wait
done
wait  # 等待最后一批
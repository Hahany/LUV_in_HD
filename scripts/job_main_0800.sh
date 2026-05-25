#!/bin/bash

#!/bin/bash
#for ((j=1; j<=1; j+=1))
#do
#  R=$(echo "scale=2; $j*0.5+1.5" | bc)
#  Rc=$(printf "%.02f" $R)
#  echo "Rc=${Rc}"
#  /usr/local/bin/python3.11 ../apps/main.py ${Rc} 100000 1000 0.800  # for main.py
#done
#

## parallel
# before run this, run previous for loop
for ((j=1; j<=140; j+=1)); do
  echo "scale=2; $j*0.5+1.5" | bc
done | parallel --line-buffer -j 20 'R={}; Rc=$(printf "%.02f" $R);ulimit -v 10000000; nice -n 19  /usr/local/bin/python3.11 ../apps/main.py ${Rc} 100000 1000 0.800'
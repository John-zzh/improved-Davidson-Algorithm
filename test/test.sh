#!/bin/bash


python ../source/Davidson.py -x methanol.xyz -b def2-SVP -m RKS -f pbe0 \
-n 5 -df True -TDA true -it 1e-3 -pt 1e-2 -o 0 1 -v 3 -chk True \
-TV 40 -max 35 

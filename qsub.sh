#$ -l tmem=8G
#$ -l gpu=true
#$ -S /bin/bash
#$ -j y
#$ -wd /cluster/project2/Neuroblastoma/MedSegBaselines

~/anaconda3/envs/env2021/bin/python Run_test.py
#$ -l tmem=16G
#$ -l gpu=true,gpu_type=(p100|v100)
#$ -S /bin/bash
#$ -j y
#$ -l h_rt=48:00:00
#$ -wd /SAN/medic/PerceptronHead/codes/SatsumaSeg/exps

~/anaconda3/envs/env2021/bin/python base02.py
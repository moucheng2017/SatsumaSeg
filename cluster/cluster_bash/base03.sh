#$ -l tmem=16G
#$ -l gpu=true,gpu_type=(titanxp|titanx)
#$ -S /bin/bash
#$ -j y
#$ -l h_rt=120:00:00
#$ -wd /SAN/medic/PerceptronHead/codes/SatsumaSeg/exps_airway

~/miniconda3/envs/pytorch1.4/bin/python base01.py
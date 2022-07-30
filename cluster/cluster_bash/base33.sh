#$ -l tmem=32G
#$ -l gpu=true,gpu_type=(a100|a100_80)
#$ -S /bin/bash
#$ -j y
#$ -l h_rt=504:00:00
#$ -wd /SAN/medic/PerceptronHead/codes/SatsumaSeg/exps_airway

~/miniconda3/envs/pytorch1.4/bin/python base33.py
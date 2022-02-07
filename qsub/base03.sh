#$ -l tmem=16G
#$ -l gpu=true,gpu_type=(titanxp|titanx)
#$ -S /bin/bash
#$ -j y
#$ -l h_rt=48:00:00
#$ -wd /SAN/medic/PerceptronHead/codes/SatsumaSeg/exps

~/miniconda3/envs/pytorch1.4/bin/python base03.py
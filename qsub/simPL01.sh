#$ -l tmem=16G
#$ -l gpu=true,gpu_type=rtx8000
#$ -S /bin/bash
#$ -j y
#$ -l h_rt=48:00:00
#$ -wd /SAN/medic/PerceptronHead/codes/SatsumaSeg/exps

~/anaconda3/envs/env2021/bin/python simPL01.py
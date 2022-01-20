#$ -l tmem=12G
#$ -l gpu=true
#$ -S /bin/bash
#$ -j y
#$ -l h_rt=30:00:00
#$ -wd /SAN/medic/PerceptronHead/codes/SatsumaSeg/exps

~/anaconda3/envs/env2021/bin/python simPL02.py
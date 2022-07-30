#$ -l tmem=16G
#$ -l gpu=true,gpu_type=(v100|rtx6000)
#$ -S /bin/bash
#$ -j y
#$ -l h_rt=334:00:00
#$ -wd /SAN/medic/PerceptronHead/codes/SatsumaSeg/exps_airway

~/miniconda3/envs/pytorch1.4/bin/python base26.py
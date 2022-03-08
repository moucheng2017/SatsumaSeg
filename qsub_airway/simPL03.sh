#$ -l tmem=16G
#$ -l gpu=true,gpu_type=(rtx8000)
#$ -S /bin/bash
#$ -j y
#$ -l h_rt=96:00:00
#$ -wd /SAN/medic/PerceptronHead/codes/SatsumaSeg/exps_airway

~/miniconda3/envs/pytorch1.4/bin/python simPL03.py
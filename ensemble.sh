#$ -l tmem=16G
#$ -l gpu=true,gpu_type=(titanxp|titanx)
#$ -S /bin/bash
#$ -j y
#$ -l h_rt=24:00:00
#$ -wd /SAN/medic/PerceptronHead/codes/SatsumaSeg/analysis

~/miniconda3/envs/pytorch1.4/bin/python Inference2DOrthogonalEnsemble.py
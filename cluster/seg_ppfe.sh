#$ -l tmem=16G
#$ -l gpu=true,gpu_type=(rtx8000|a100)
#$ -S /bin/bash
#$ -j y
#$ -l h_rt=4:00:00
#$ -wd /SAN/medic/PerceptronHead/codes/SatsumaSeg/inference/inference_ppfe/

~/miniconda3/envs/pytorch1.4/bin/python run_inference.py
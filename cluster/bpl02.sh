#$ -l tmem=16G
#$ -l gpu=true,gpu_type=(rtx8000|a100_80)
#$ -S /bin/bash
#$ -j y
#$ -l h_rt=168:00:00
#$ -wd /SAN/medic/PerceptronHead/codes/SatsumaSeg/

~/miniconda3/envs/pytorch1.4/bin/python Run.py --data '/SAN/medic/PerceptronHead/data/lung/private/airway' --log_tag 'exp' --iterations 5000 --lr 0.01 --depth 3 --width 16 --batch 4 --unlabelled 2 --mu 0.8 --sigma 0.1 --full_orthogonal 0 --zoom 1 --sampling 5 --new_size_h 384 --new_size_w 384 --detach 1
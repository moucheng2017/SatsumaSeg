#$ -l tmem=16G
#$ -l gpu=true,gpu_type=(p100|v100|rtx6000|rtx8000)
#$ -S /bin/bash
#$ -j y
#$ -l h_rt=168:00:00
#$ -wd /SAN/medic/PerceptronHead/codes/SatsumaSeg/

~/miniconda3/envs/pytorch1.4/bin/python Run.py --data '/SAN/medic/PerceptronHead/data/lung/private/airway' --log_tag 'exp' --format 'nii' --iterations 832000 --lr 1e-4 --depth 4 --width 32 --batch 8 --full_orthogonal 1 --new_size_h 480 --new_size_w 480 --patience 16640 --ema_saving_starting 166400 --unlabelled 4 --warmup_start 33280
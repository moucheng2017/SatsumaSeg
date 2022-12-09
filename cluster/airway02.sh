#$ -l tmem=16G
#$ -l gpu=true,gpu_type=(rtx8000|a100_80)
#$ -S /bin/bash
#$ -j y
#$ -l h_rt=168:00:00
#$ -wd /SAN/medic/PerceptronHead/codes/SatsumaSeg/

~/miniconda3/envs/pytorch1.4/bin/python Run.py --data '/SAN/medic/PerceptronHead/data/airway' --log_tag 'airway' --format 'nii' --iterations 832000 --lr 1e-4 --depth 4 --width 32 --batch 4 --full_orthogonal 1 --new_size_h 448 --new_size_w 448 --patience 16640 --ema_saving_starting 166400 --unlabelled 2 --warmup_start 33280
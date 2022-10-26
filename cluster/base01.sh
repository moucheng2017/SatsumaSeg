#$ -l tmem=16G
#$ -l gpu=true,gpu_type=(rtx6000|rtx8000|v100)
#$ -S /bin/bash
#$ -j y
#$ -l h_rt=168:00:00
#$ -wd /SAN/medic/PerceptronHead/codes/SatsumaSeg/

~/miniconda3/envs/pytorch1.4/bin/python Run_Sup.py
--data '/SAN/medic/PerceptronHead/data/lung/private/airway'
--log_tag 'exp'
--iterations 5000
--lr 0.005
--depth 3
--width 16
--batch 4
--full_orthogonal 0
--zoom 1
--new_size_h 384
--new_size_w 384
--saving_starting 1000
--saving_frequency 100
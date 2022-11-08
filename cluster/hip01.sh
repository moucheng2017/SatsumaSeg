#$ -l tmem=16G
#$ -l gpu=true,gpu_type=!(gtx1080ti|rtx2080ti)
#$ -S /bin/bash
#$ -j y
#$ -l h_rt=24:00:00
#$ -wd /SAN/medic/PerceptronHead/codes/SatsumaSeg/

~/miniconda3/envs/pytorch1.4/bin/python RunHip.py --data '/SAN/medic/PerceptronHead/data/hipct/hip_covid' --log_tag 'exp' --iterations 5000 --lr 0.01 --depth 4 --width 16 --batch 1 --unlabelled 1 --mu 0.9 --sigma 0.1 --full_orthogonal 0 --zoom 1 --sampling 5 --new_size_h 512 --new_size_w 512 --detach 0 --saving_starting 50

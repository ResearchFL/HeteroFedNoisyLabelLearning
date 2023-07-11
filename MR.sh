# nohup bash run_main.sh >> _p.log 2>&1 &

level_n=1
gpu_num=1
begin_sel_r=5

python main.py -alg MR --dataset cifar10 --model resnet18 --rounds2 200 --lr 0.1 --plr 0.1 --num_users 100 --frac2 0.1 --begin_sel $begin_sel_r --gpu $gpu_num --level_n_system $level_n

# python main.py -alg MR --dataset mnist --model resnet18 --rounds2 200 --lr 0.1 --plr 0.1 --num_users 100 --frac2 0.1 --begin_sel 5 --gpu 1 --level_n_system 1

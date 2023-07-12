# nohup bash MR.sh >> ./record/eminist_minist_MR.log 2>&1 &

level_n=1
gpu_num=1
begin_sel_r=5

#python -u main.py -alg MR --dataset mnist --model lenet --rounds2 200 --lr 0.1 --num_users 100 --frac2 0.1 --gpu $gpu_num --level_n_system $level_n
#python -u main.py -alg MR --dataset cifar10 --model resnet18 --rounds2 450 --lr 0.01 --num_users 100 --frac2 0.1 --gpu $gpu_num --level_n_system $level_n
#python -u main.py -alg MR --dataset cifar100 --model resnet34 --rounds2 450 --lr 0.01 --num_users 50 --frac2 0.2 --gpu $gpu_num --level_n_system $level_n
#
#python -u main.py -alg FedAVG --dataset mnist --model lenet --rounds2 200 --lr 0.1 --num_users 100 --frac2 0.1 --gpu $gpu_num --level_n_system $level_n
#python -u main.py -alg FedAVG --dataset cifar10 --model resnet18 --rounds2 450 --lr 0.01 --num_users 100 --frac2 0.1 --gpu $gpu_num --level_n_system $level_n
#python -u main.py -alg FedAVG --dataset cifar100 --model resnet34 --rounds2 450 --lr 0.01 --num_users 50 --frac2 0.2 --gpu $gpu_num --level_n_system $level_n

#python main.py -alg MR --dataset clothing1m --model resnet50 --rounds2 50 --lr 0.001 --num_users 300 --frac2 0.03 --gpu $gpu_num --level_n_system $level_n
#python main.py -alg MR --dataset mnist --model resnet18 --rounds2 200 --lr 0.1 --plr 0.1 --num_users 100 --frac2 0.1 --begin_sel 5 --gpu 1 --level_n_system 1

python -u main.py -alg MR2 --dataset mnist --model lenet --rounds2 200 --lr 0.1 --num_users 100 --frac2 0.1 --gpu $gpu_num --level_n_system $level_n

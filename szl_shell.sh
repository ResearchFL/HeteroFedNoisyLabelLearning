# nohup bash szl_shell.sh &

run_times=5
level_n=1
gpu_num=1
begin_sel_r=10

#python -u main.py -alg MR --dataset mnist --model lenet --rounds2 200 --lr 0.1 --num_users 100 --frac2 0.1 --gpu $gpu_num --level_n_system $level_n
#python -u main.py -alg MR --dataset cifar10 --model resnet18 --rounds2 450 --lr 0.01 --num_users 100 --frac2 0.1 --gpu $gpu_num --level_n_system $level_n
#python -u main.py -alg MR --dataset cifar100 --model resnet34 --rounds2 450 --lr 0.01 --num_users 50 --frac2 0.2 --gpu $gpu_num --level_n_system $level_n
#
#python -u main.py -alg FedAVG --dataset mnist --model lenet --rounds2 200 --lr 0.1 --num_users 100 --frac2 0.1 --gpu $gpu_num --level_n_system $level_n
#python -u main.py -alg FedAVG --dataset cifar10 --model resnet18 --rounds2 450 --lr 0.01 --num_users 100 --frac2 0.1 --gpu $gpu_num --level_n_system $level_n
#python -u main.py -alg FedAVG --dataset cifar100 --model resnet34 --rounds2 450 --lr 0.01 --num_users 50 --frac2 0.2 --gpu $gpu_num --level_n_system $level_n

#python main.py -alg MR --dataset clothing1m --model resnet50 --rounds2 50 --lr 0.001 --num_users 300 --frac2 0.03 --gpu $gpu_num --level_n_system $level_n
#python main.py -alg MR --dataset mnist --model resnet18 --rounds2 200 --lr 0.1 --plr 0.1 --num_users 100 --frac2 0.1 --begin_sel 5 --gpu 1 --level_n_system 1

#python -u main.py -alg MR --dataset mnist --model lenet --rounds2 200 --lr 0.1 --num_users 100 --frac2 0.1 --gpu $gpu_num --level_n_system $level_n

'''
for((c = 1; c < run_times + 1; c ++))
do
python main.py -alg FedTwin --dataset mnist --model lenet --rounds2 200 --lr 0.1 --plr 0.1 --num_users 100 --frac2 0.1 --begin_sel $begin_sel_r --gpu $gpu_num --level_n_system $level_n
done
'''

#python -u main.py -alg FedAVG --dataset mnist --model lenet --rounds2 200 --lr 0.1 --num_users 100 --frac2 0.1 --gpu $gpu_num --level_n_system $level_n
#python -u main.py -alg FedAVG --dataset cifar10 --model resnet18 --rounds2 450 --lr 0.01 --num_users 100 --frac2 0.1 --gpu $gpu_num --level_n_system $level_n
#python -u main.py -alg FedAVG --dataset cifar100 --model resnet34 --rounds2 450 --lr 0.01 --num_users 50 --frac2 0.2 --gpu $gpu_num --level_n_system $level_n

currentdate=$(date +%Y%m%d)


mkdir ./record/"${currentdate}"
mkdir ./record/"${currentdate}"/MR ./record/"${currentdate}"/FedAVG ./record/"${currentdate}"/FedTwin
touch ./record/"${currentdate}"/MR/MR_minist.log ./record/"${currentdate}"/MR/MR_cifar10.log ./record/"${currentdate}"/MR/MR_cifar100.log
touch ./record/"${currentdate}"/FedAVG/FedAVG_minist.log ./record/"${currentdate}"/FedAVG/FedAVG_cifar10.log ./record/"${currentdate}"/FedAVG/FedAVG_cifar100.log
touch ./record/"${currentdate}"/FedTwin/FedTwin_minist.log ./record/"${currentdate}"/FedTwin/FedTwin_cifar10.log ./record/"${currentdate}"/FedTwin/FedTwin_cifar100.log

for((c = 1; c < run_times + 1; c ++))
do

python -u main.py -alg FedTwin --dataset mnist --model lenet --rounds2 200 --lr 0.1 --plr 0.1 --num_users 100 --frac2 0.1 \
--begin_sel $begin_sel_r --gpu $gpu_num --level_n_system $level_n >> ./record/"${currentdate}"/FedTwin/FedTwin_minist.log 2>&1

python main.py -alg FedTwin --dataset cifar10 --model resnet18 --rounds2 450 --lr 0.01 --plr 0.01 --num_users 100 --frac2 0.1 \
--begin_sel $begin_sel_r --gpu $gpu_num --level_n_system $level_n >> ./record/"${currentdate}"/FedTwin/FedTwin_cifar10.log 2>&1

python main.py -alg FedTwin --dataset cifar100 --model resnet34 --rounds2 450 --lr 0.01 --plr 0.01 --num_users 50 --frac2 0.2 \
--begin_sel $begin_sel_r --gpu $gpu_num --level_n_system $level_n >> ./record/"${currentdate}"/FedTwin/FedTwin_cifar100.log 2>&1

python -u main.py -alg MR --dataset mnist --model lenet --rounds2 200 --lr 0.1 --num_users 100 --frac2 0.1 --gpu $gpu_num \
--level_n_system $level_n >> ./record/"${currentdate}"/MR/MR_minist.log 2>&1

python -u main.py -alg MR --dataset cifar10 --model resnet18 --rounds2 450 --lr 0.01 --num_users 100 --frac2 0.1 --gpu $gpu_num \
--level_n_system $level_n >> ./record/"${currentdate}"/MR/MR_cifar10.log 2>&1

python -u main.py -alg MR --dataset cifar100 --model resnet34 --rounds2 450 --lr 0.01 --num_users 50 --frac2 0.2 --gpu $gpu_num \
--level_n_system $level_n >> ./record/"${currentdate}"/MR/MR_cifar100.log 2>&1

python -u main.py -alg FedAVG --dataset mnist --model lenet --rounds2 200 --lr 0.1 --num_users 100 --frac2 0.1 --gpu $gpu_num \
--level_n_system $level_n >> ./record/"${currentdate}"/FedAVG/FedAVG_minist.log 2>&1

python -u main.py -alg FedAVG --dataset cifar10 --model resnet18 --rounds2 450 --lr 0.01 --num_users 100 --frac2 0.1 --gpu $gpu_num \
--level_n_system $level_n >> ./record/"${currentdate}"/FedAVG/FedAVG_cifar10.log 2>&1

python -u main.py -alg FedAVG --dataset cifar100 --model resnet34 --rounds2 450 --lr 0.01 --num_users 50 --frac2 0.2 --gpu $gpu_num \
--level_n_system $level_n >> ./record/"${currentdate}"/FedAVG/FedAVG_cifar100.log 2>&1

done
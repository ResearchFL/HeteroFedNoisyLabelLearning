
# nohup bash run_experiments.sh >> ./result/${date}.log 2>&1 &


# fraction of noisy clients
level_n=1
gpu_num=0
begin_sel_r=5
for((c = 1; c < 2; c ++))
do
	#fedvag local only
		python main.py -alg FedTwin --dataset mnist --model lenet --rounds2 200 --lr 0.1 --plr 0.1 --num_users 100 --frac2 0.1 --begin_sel $begin_sel_r --gpu $gpu_num --level_n_system $level_n
		python main.py -alg RFL --dataset mnist --model lenet --rounds2 200 --lr 0.1 --num_users 100 --frac2 0.1 --gpu $gpu_num
		python main.py -alg FedTwin --dataset cifar10 --model resnet18 --rounds2 450 --lr 0.01 --plr 0.01 --num_users 100 --frac2 0.1 --begin_sel $begin_sel_r --gpu $gpu_num --level_n_system $level_n
		python main.py -alg RFL --dataset cifar10 --model resnet18 --rounds2 450 --lr 0.01 --num_users 100 --frac2 0.1 --gpu $gpu_num --level_n_system $level_n
		python main.py -alg FedTwin --dataset cifar100 --model resnet34 --rounds2 450 --lr 0.01 --plr 0.01 --num_users 50 --frac2 0.2 --begin_sel $begin_sel_r --gpu $gpu_num --level_n_system $level_n
		python main.py -alg RFL --dataset cifar100 --model resnet34 --rounds2 450 --lr 0.01 --num_users 50 --frac2 0.2 --gpu $gpu_num --level_n_system $level_n
		python main.py -alg FedTwin --dataset clothing1m --model resnet50 --rounds2 50 --lr 0.001 --plr 0.001 --num_users 300 --frac2 0.03 --begin_sel $begin_sel_r --gpu $gpu_num --level_n_system $level_n
		python main.py -alg RFL --dataset clothing1m --model resnet50 --rounds2 50 --lr 0.001 --num_users 300 --frac2 0.03 --gpu $gpu_num --level_n_system $level_n


		python main.py -alg FedCorr --dataset mnist --model lenet --rounds2 200 --lr 0.1 --num_users 100 --frac2 0.1 --gpu $gpu_num --level_n_system $level_n
		python main.py -alg FedCorr --dataset cifar10 --model resnet18 --rounds2 450 --lr 0.01 --num_users 100 --frac2 0.1 --gpu $gpu_num --level_n_system $level_n
		python main.py -alg FedCorr --dataset cifar100 --model resnet34 --rounds2 450 --lr 0.01 --num_users 50 --frac2 0.2 --gpu $gpu_num --level_n_system $level_n
    python main.py -alg FedCorr --dataset clothing1m --model resnet50 --rounds2 50 --lr 0.001 --num_users 300 --frac2 0.03 --gpu $gpu_num --level_n_system $level_n

done

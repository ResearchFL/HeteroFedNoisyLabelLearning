
# 脚本运行 nohup bash run_fedtwin12.sh &
run_times=1
gpu_num=3
begin_sel_r=10

#文件夹创建函数
MakeDir(){
    if ! [ -e "$1" ]
    then
        mkdir "$1"
    fi
}

#数据集
listDataset=(mnist cifar10 cifar100 clothing1m)
#模型---对应数据集
listModel=(lenet resnet18 resnet34 renet50)
#轮次---对应数据集
listRound=(30 450 450 50)
#客户端数量---对应数据集
listClient=(100 100 50 300)
#frac2---对应数据集
listFrac2=(0.1 0.1 0.2 0.3)
#lr---对应数据集
listLr=(0.1 0.01 0.01 0.001)

#方法
listMethod=(FedTwin1 FedTwin2)

#定义ρ和τ
listRou=(0.0 0.5 1)
listTau=(0.0 0.3 0.5)

#是否IID
listIID=(0 1)

#获取当前日期
currentDate="$(date +%Y%m%d)"

#创建以当前日期为名称的文件夹
date_path=./record/"${currentDate}"
MakeDir ./record
MakeDir "${date_path}"

#开始训练
for ((i=0;i<1;i++))    #遍历数据集
do
    for ((time=1;time<="${run_times}";time++))
    do
        for ((j=0;j<${#listMethod[@]};j++))        #遍历方法
        do
            for ((p=1;p<=1;p++))            #遍历IID情况1
            do
                for ((q=1;q<=1;q++))            #遍历Rou和Tau1
                do
                    case ${listIID[p]} in

                    "0")                    #IID

                    logFile="$date_path"/"${listDataset[i]}"_"${listMethod[j]}"_IID_rou_"${listRou[q]}"_tau_"${listTau[q]}"_"${i}".log  #文件路径

                    touch "${logFile}"

                    python -u main.py --alg "${listMethod[j]}" --dataset "${listDataset[i]}" --model "${listModel[i]}" --rounds2 "${listRound[i]}" --num_users "${listClient[i]}" \
                        --lr "${listLr[i]}" --plr "${listLr[i]}" --frac2 "${listFrac2[i]}" \
                        --begin_sel $begin_sel_r --gpu $gpu_num \
                        --level_n_system "${listRou[q]}" --level_n_lowerb "${listTau[q]}" \
                        --iid \
                        >> "${logFile}" 2>&1
                    ;;

                    "1")                    #NonIID

                    logFile="$date_path"/"${listDataset[i]}"_"${listMethod[j]}"_nonIID_rou_"${listRou[q]}"_tau_"${listTau[q]}"_"${i}".log  #文件路径

                    touch "${logFile}"

                    python -u main.py --alg "${listMethod[j]}" --dataset "${listDataset[i]}" --model "${listModel[i]}" --rounds2 "${listRound[i]}" --num_users "${listClient[i]}"  \
                        --lr "${listLr[i]}" --plr "${listLr[i]}" --frac2 "${listFrac2[i]}" \
                        --begin_sel $begin_sel_r --gpu $gpu_num \
                        --level_n_system "${listRou[q]}" --level_n_lowerb "${listTau[q]}" \
                        \
                        >> "${logFile}" 2>&1
                    ;;
                    esac
                done
            done
        done
    done
done




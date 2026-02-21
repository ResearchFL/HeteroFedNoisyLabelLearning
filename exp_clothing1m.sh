
# 脚本运行 nohup bash exp_clothing1m.sh &
run_times=3
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
#GPU-- 对应数据集
gpu_nums=(4 5)
#模型---对应数据集
listModel=(lenet resnet18 resnet34 renet50)
#轮次---对应数据集
listRound=(200 450 450 50)
#客户端数量---对应数据集
listClient=(100 100 50 300)
#frac2---对应数据集
listFrac2=(0.1 0.1 0.2 0.03)
#lr---对应数据集
listLr=(0.1 0.01 0.01 0.001)

#方法
listMethod=(FedTwin FedCorr RFL FedAVG MR FedProx)

#定义ρ和τ
listRou=(1)
listTau=(0.3)

#是否IID
listIID=(1)

#获取当前日期
currentDate="$(date +%Y%m%d)"

#创建以当前日期为名称的文件夹
date_path=./record/"${currentDate}"_clothing1m
MakeDir ./record
MakeDir "${date_path}"

gpu=0
bingxing=0

#开始训练
for ((i=3;i<4;i++))    #遍历数据集
do
    for ((time=1;time<="${run_times}";time++))
    do
        for ((j=0;j<${#listMethod[@]};j++))        #遍历方法
        do
            for ((p=0;p<${#listIID[@]};p++))            #遍历IID情况
            do
                for ((q=0;q<${#listRou[@]};q++))            #遍历Rou和Tau
                do
                    case ${listIID[p]} in

                    "0")                    #IID

                    logFile="$date_path"/"${listDataset[i]}"_"${listMethod[j]}"_IID_rou_"${listRou[q]}"_tau_"${listTau[q]}".log  #文件路径

                    touch "${logFile}"

                    python -u main.py --alg "${listMethod[j]}" --dataset "${listDataset[i]}" --model "${listModel[i]}" --rounds2 "${listRound[i]}" --num_users "${listClient[i]}" \
                        --lr "${listLr[i]}" --plr "${listLr[i]}" --frac2 "${listFrac2[i]}" \
                        --begin_sel $begin_sel_r --gpu  ${gpu_nums[gpu]} \
                        --level_n_system "${listRou[q]}" --level_n_lowerb "${listTau[q]}" \
                        --iid \
                        >> "${logFile}" 2>&1 &
                    ;;

                    "1")                    #NonIID

                    logFile="$date_path"/"${listDataset[i]}"_"${listMethod[j]}"_nonIID_rou_"${listRou[q]}"_tau_"${listTau[q]}".log  #文件路径

                    touch "${logFile}"

                    python -u main.py --alg "${listMethod[j]}" --dataset "${listDataset[i]}" --model "${listModel[i]}" --rounds2 "${listRound[i]}" --num_users "${listClient[i]}"  \
                        --lr "${listLr[i]}" --plr "${listLr[i]}" --frac2 "${listFrac2[i]}" \
                        --begin_sel $begin_sel_r --gpu  ${gpu_nums[gpu]} \
                        --level_n_system "${listRou[q]}" --level_n_lowerb "${listTau[q]}" \
                        \
                        >> "${logFile}" 2>&1 &
                    ;;
                    esac

                    let "bingxing=bingxing+1"
                    let "gpu=(gpu+1)%2"
                    echo $gpu
                    echo $bingxing

                    if [ ${bingxing%4} -eq 0 ]; then
                        bingxing=0
                        wait
                    fi
                done
            done
#            bingxing=$((bingxing+1))
#            gpu=$(((gpu+1)%2))
#            if (bingxing % 4 == 0)
#            then
#                wait
#                bingxing=0
#            fi
        done
    done
done

wait


hostnames=("orichalcum.liacs.nl" "dimeritium.liacs.nl" "carbonite.liacs.nl" "vibranium.liacs.nl")
script="./exp.sh"

for domain in "${hostnames[@]}"
do
  # 在每个域名上运行脚本文件
  ssh jij@"$domain" "bash \$script" &
done

wait
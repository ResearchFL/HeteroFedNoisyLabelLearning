import os
import re
import numpy as np



# 定义数据集、算法、rho 和 tau 的值
datasets = ['mnist', 'cifar10', 'cifar100']
methods = ['LocalCORES', 'LocalKNN', 'GlobalCORES', 'GlobalKNN', 'FedAVG', 'FedProx', 'FedCorr', 'RFL', 'MR', 'FedTwin']
datasets_label = ['MNIST', 'CIFAR-10', 'CIFAR-100']
methods_label = ['Local + CORES', 'Local + KNN', 'Global + CORES', 'Global + KNN', 'FedAvg', 'FedProx', 'FedCorr', 'RFL', 'MR', 'FedTwin']
rhos = [0.0, 0.5, 1]
taus = [0.0, 0.5, 1]
taus_true = [0.0, 0.3, 0.5]
Rounds = [200, 450, 450, 50]
IID_or_not = ['IID', 'nonIID']

# 创建一个空的表格
table = np.empty((len(datasets), len(methods), len(IID_or_not), len(rhos)), dtype=object)
table.fill("~")

# 遍历 log 文件
basePath = "./"
file_list = os.listdir(basePath)
for filename in file_list:
    if filename.endswith('.log'):
        # 解析文件名获取参数值
        # match = re.match(r'(\w+)_(\w+)_(\w+)_rou_(\d+)_tau_(\d+)\.log', filename)
        # if match:
        #     dataset, method, iid, rho, tau = match.groups()
        match = re.match(r'(\w+)_(\w+)_(\w+)_rou_(\d+\.?\d*)_tau_(\d+\.?\d*)\.log', filename)

        # match = re.match(r'(\w+)_(\w+)_(\w+)_(\d+\.\d+)_tau_(\d+\.\d+)\.log', filename)
        if match:
            dataset, method, iid, rho, tau = match.groups()
            if iid is None:
                iid = ''  # 将 iid 设置为空字符串或其他默认值
            # 确定参数在表格中的索引
            dataset_idx = datasets.index(dataset)
            method_idx = methods.index(method)
            rho_idx = rhos.index(float(rho))
            iid_idx = IID_or_not.index(iid)
            # 读取 log 文件内容并计算平均值
            with open(basePath+filename, 'r') as file:
                contents = file.read()
                # start = contents.find(f"Round {Rounds[dataset_idx]-1} global test acc")
                # if start != -1:
                #     end = contents.find("\n", start)
                #     test_acc = contents[start:end].strip()
                #     values = test_acc.split[5]
                # else:
                #     print("not find")
                values = re.findall(r'Round {} global test acc  (\d+\.\d+)'.format(Rounds[dataset_idx]-1), contents)
                if values:
                    average = np.mean([float(value) for value in values])
                    std = np.std([float(value) for value in values], ddof=1)
                    table[dataset_idx, method_idx, iid_idx, rho_idx] = f"{average:.2f} $\pm$ {std:.2f}"
        else:
            print(filename)
print('\n')
# 打印表格的 LaTeX 代码
print("\\begin{table*}[!ht]")
print("    \\centering")
print("    \\\scalebox{0.9}{")
print("    \\begin{tabular}{c|l|ccc|ccc}")
print("    \\hline")
print("        \\multirow{2}{*}{Datasets} & \\multirow{2}{*}{Methods} & \\multicolumn{3}{c|}{\\multirow{2}{*}{IID}} & \\multicolumn{3}{c}{\\multirow{2}{*}{nonIID}} \\\\")
print("        &  &  &  &  &  &  &  \\\\ ")
print("        \\hline")
print("        &  & $\\rho$ = 0.0 & $\\rho$  = 0.5 & $\\rho$  = 1 & $\\rho$ = 0.0 & $\\rho$ = 0.5 & $\\rho$ = 1 \\\\ ")
print("        &  & $\\tau$ = 0.0 & $\\tau$ = 0.3 & $\\tau$ = 0.5 & $\\tau$ = 0.0 & $\\tau$ = 0.3 & $\\tau$ = 0.5 \\\\ \\hline")

# 输出表格内容
for i, dataset in enumerate(datasets_label):
    for j, method in enumerate(methods_label):
        if method == "Local + CORES":
            row = f"\\multirow{{12}}{{*}}{{{dataset}}} & {method}"
        else:
            row = f"~ & {method}"
        for l, iid in enumerate(IID_or_not):
            for k, rho in enumerate(rhos):
                # for l, tau in enumerate(taus):
                value = table[i, j, l, k]
                row += f" & {value}"
        row += " \\\\"
        print(row)
    print("\\hline\\hline")

print("\\end{tabular}")
print("\\}")
print("\\end{table*}")
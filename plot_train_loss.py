import numpy as np
import matplotlib.pyplot as plt

# 从文本文件中读取测试准确率数据
with open('./record/FedTwin/mnist_lenet_NL_0.4_LB_0.5_Rnd_300_ep_5_Frac_0.10_LR_0.100_Seed_13_nonIID_p_0.7_dirich_10.0_15_1_acc.txt', 'r') as file:
    lines = file.readlines()

# 提取测试准确率
test_acc = []
for line in lines:
    if 'train loss' in line:
        acc = float(line.split()[-1])
        test_acc.append(acc)

# 生成轮数
rounds = np.arange(len(test_acc))

# 绘制曲线图
plt.plot(rounds, test_acc, marker='o')
plt.xlabel('Round')
plt.ylabel('Train Loss')
plt.title('Train Loss vs. Round')
plt.xticks(rounds)
plt.grid(True)
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# 从文本文件中读取测试准确率数据
with open('./record/cifar10_adjust/aaa.txt', 'r') as file:
    lines = file.readlines()

# # 提取测试准确率
# test_acc = []
# train_acc = []
# for line in lines:
#     if 'global train acc' in line:
#         acc = float(line.split()[-1])
#         train_acc.append(acc)
# 提取测试准确率
test_acc = []
for line in lines:
    if 'global test acc' in line:
        acc = float(line.split()[-1])
        test_acc.append(acc)

# # 提取训练准确率
# train_acc = []
# for line in lines:
#     if 'global train acc' in line:
#         acc = float(line.split()[-1])
#         train_acc.append(acc)
# 生成轮数
rounds = np.arange(len(test_acc))

# 绘制曲线图
plt.plot(rounds, test_acc, marker='o')
# plt.plot([i for i in range(0, len(rounds), 50)], [i for i in range(0, len(test_acc), 50)], marker='o')

# plt.plot(rounds, train_acc, marker='o')
plt.xlabel('Round')
plt.ylabel('Test Accuracy')
plt.title('Test Accuracy vs. Round')
plt.xticks([i for i in range(0, len(rounds), 50)])
plt.grid(True)
plt.savefig('./record/cifar10_adjust/FedFixer.pdf', format='pdf', bbox_inches='tight')
plt.show()
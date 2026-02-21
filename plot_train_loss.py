import numpy as np
import matplotlib.pyplot as plt
import re
# 从文本文件中读取测试准确率数据
with open('./record/cifar10_adjust/bbb.txt', 'r') as file:
    lines = file.readlines()

# 提取测试准确率
test_acc = []
for line in lines:
    if re.search(r'Round \d+ train loss', line):
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

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_circles
from sklearn.preprocessing import StandardScaler

# 生成数据
X, y = make_circles(n_samples=100, factor=0.3, noise=0.1, random_state=42)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
y = y.astype(np.float32)

# 转换为PyTorch张量
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)


# 定义全连接神经网络模型
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(2, 10)
        self.fc2 = nn.Linear(10, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return self.softmax(x)


# 初始化模型、损失函数和优化器
model = MLP()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 训练模型
for epoch in range(1000):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()

# 生成均匀分布的坐标点
x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
grid_points = np.c_[xx.ravel(), yy.ravel()]
grid_tensor = torch.tensor(grid_points, dtype=torch.float32)

# 预测softmax概率
model.eval()
with torch.no_grad():
    probs = model(grid_tensor).numpy()
prob_label_1 = probs[:, 1]  # Assuming label 1 is of interest

# 找到概率小于等于0.7的点
mask_0_7 = prob_label_1 <= 0.7
Z_0_7 = mask_0_7.reshape(xx.shape)

# 找到概率等于0.5的点的轮廓
mask_0_5 = prob_label_1 <= 0.5 # 允许一个小的容差
Z_0_5 = mask_0_5.reshape(xx.shape)

# 画图
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z_0_7, levels=[0, 0.5, 1], colors=['red', 'blue'], alpha=0.3, hatches=['.', '+'])
contour = plt.contour(xx, yy, Z_0_5, levels=[0,0.5, 1], colors=['black', 'orange'], linestyles='--')
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap='bwr', marker='o', edgecolor='k')
plt.title('Training Data and Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()


# import matplotlib.pyplot as plt
# import numpy as np
#
# # 创建数据
# x = np.linspace(-3, 3, 100)
# y = np.linspace(-3, 3, 100)
# X, Y = np.meshgrid(x, y)
# Z = np.sin(X) * np.cos(Y)
#
# # 绘制图形
# plt.figure(figsize=(8, 6))
# contourf = plt.contourf(X, Y, Z, levels=10, cmap='coolwarm')
#
# # 添加图案
# hatches = ['/', '\\', '|', '-', '+', '.']
# for i, collection in enumerate(contourf.collections):
#     collection.set_hatch(hatches[i % len(hatches)])
#
# plt.colorbar(contourf)
# plt.title('Contour plot with hatching')
# plt.show()

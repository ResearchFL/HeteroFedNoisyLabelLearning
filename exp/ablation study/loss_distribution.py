import matplotlib.pyplot as plt
import ast
from matplotlib.ticker import ScalarFormatter, MaxNLocator
from matplotlib.font_manager import FontProperties
import matplotlib as mpl

# 设置字体为 Times New Roman 和字体大小为 20
font = FontProperties(family='Times New Roman', size=25, weight='bold')
font1 = FontProperties(family='Times New Roman', size=20, weight='bold')
font2 = FontProperties(family='Times New Roman', size=16, weight='bold')
# 设置全局字体样式
mpl.rcParams['font.family'] = 'Times New Roman'
plt.rc('font', size=20, weight='bold')

# 定义一个函数来读取列表数据
def read_list_from_log(file_path, line_number):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        if line_number <= len(lines):
            line = lines[line_number - 1]
            try:
                # 将字符串转换为列表对象
                list_object = ast.literal_eval(line)
                return list_object
            except ValueError:
                return "无法解析为列表。"
        else:
            return "行号超出文件范围。"

# 从日志文件中读取数据
clean_loss = read_list_from_log(
    "./cifar10_FedTwin_wihout_CR_IID_rou_0.5_tau_0.3.log",
    215
)
noisy_loss = read_list_from_log(
    "./cifar10_FedTwin_wihout_CR_IID_rou_0.5_tau_0.3.log",
    217
)

# 设置直方图的边界和区间数量
bins = 50
range_min = min(min(clean_loss), min(noisy_loss))
range_max = max(max(clean_loss), max(noisy_loss))
bin_range = (range_max - range_min) / bins


# 绘制直方图
# plt.figure(figsize=(10, 6))  # 设置图形的大小

# 绘制两个直方图
plt.hist(clean_loss, bins=int(bins), range=(range_min, range_max), alpha=0.5, label='clean samples')
plt.hist(noisy_loss, bins=int(bins), range=(range_min, range_max), alpha=0.5, label='noisy samples')
plt.xlabel('Loss', fontproperties=font)  # 设置x轴标签字体大小
plt.ylabel('Number of samples', fontproperties=font)  # 设置y轴标签字体大小
plt.xticks(fontproperties=font1)  # 设置x轴刻度标签字体大小
plt.yticks(fontproperties=font1)  # 设置y轴刻度标签字体大小


# 设置纵坐标刻度显示为科学计数法
plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

# 使用ScalarFormatter来格式化刻度标签
ax = plt.gca()
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

# 设置y轴刻度范围
y_axis_min = 0
# y_axis_max = 3e3  # Set the desired upper limit of the y-axis ticks
# plt.ylim(y_axis_min, y_axis_max)

# 设置y轴刻度显示为整数
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
# 设置y轴刻度数量，最多5个
ax.yaxis.set_major_locator(MaxNLocator(nbins=4))


plt.legend(prop=font2)

# 调整坐标轴线的线宽为2
ax.spines['top'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)

# 调整坐标轴刻度线的线宽为1.5
ax.tick_params(axis='both', width=1.5)

# # 设置坐标轴刻度标签的字体为 Times New Roman 和字体大小为 20，加粗
# ax.set_xticks(ax.get_xticks())
# ax.set_yticks(ax.get_yticks())
# ax.set_xticklabels(ax.get_xticklabels(), fontproperties=font)
# ax.set_yticklabels(ax.get_yticklabels(), fontproperties=font)

# 保存图表为高清的PDF文件
plt.savefig('loss_distribution_without_CR.pdf', dpi=300, bbox_inches='tight')

plt.show()
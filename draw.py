import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置绘图风格
plt.style.use('seaborn')
sns.set(font_scale=1.2)
plt.rcParams['font.family'] = 'Times New Roman'

# ==========================
# 图5.1：不同IoU阈值下的F1分数
# ==========================

# 模拟数据 (IoU阈值: F1分数)
thresholds = [0.3, 0.5, 0.7]
f1_scores = [0.52, 0.20, 0.10]

fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.bar(thresholds, f1_scores,
              width=0.15,
              color=sns.color_palette("Blues_d", n_colors=3),
              edgecolor='black')

# 添加数据标签
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}',
            ha='center', va='bottom')

# 坐标轴设置
ax.set_xlabel('IoU Threshold', fontsize=12, labelpad=10)
ax.set_ylabel('F1 Score', fontsize=12, labelpad=10)
ax.set_xticks(thresholds)
ax.set_xticklabels([f'{t}' for t in thresholds])
ax.set_ylim(0, 0.6)
ax.grid(True, linestyle='--', alpha=0.6)

# 添加标题
plt.title('F1 Scores at Different IoU Thresholds (n=100)',
          fontsize=14, pad=20)

plt.tight_layout()
plt.savefig('figure5_1.png', dpi=300, bbox_inches='tight')
plt.close()

# ==========================
# 图5.2：IoU分布直方图
# ==========================

# 生成模拟IoU数据（正态分布截断到0-1）
np.random.seed(42)
mu, sigma = 0.35, 0.15
raw_data = np.random.normal(mu, sigma, 100)
iou_data = np.clip(raw_data, 0, 1)  # 截断到[0,1]范围

fig, ax = plt.subplots(figsize=(8, 6))

# 绘制直方图
n, bins, patches = ax.hist(iou_data, bins=10,
                           range=(0, 1),
                           color='#4B77BE',
                           edgecolor='black',
                           alpha=0.8)

# 添加频数标签
for i in range(len(patches)):
    plt.text(patches[i].get_x() + patches[i].get_width()/2,
             patches[i].get_height()+0.5,
             str(int(n[i])),
             ha='center')

# 坐标轴设置
ax.set_xlabel('Intersection over Union (IoU)', fontsize=12, labelpad=10)
ax.set_ylabel('Frequency', fontsize=12, labelpad=10)
ax.set_xlim(0, 1)
ax.set_xticks(np.linspace(0, 1, 11))
ax.grid(True, linestyle='--', alpha=0.6)

# 添加标题
plt.title('Distribution of IoU Values (n=100)',
         fontsize=14, pad=20)

plt.tight_layout()
plt.savefig('figure5_2.png', dpi=300, bbox_inches='tight')
plt.close()

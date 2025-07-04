import matplotlib.pyplot as plt
import networkx as nx
from matplotlib import rcParams

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决保存图像时负号'-'显示为方块的问题

# 创建有向图
G = nx.DiGraph()

# 定义节点和连接关系
nodes = [
    "输入图像",
    "b1: 大卷积核提取特征",
    "b2: 降维+局部特征提取",
    "b3-b4: Inception多尺度融合",
    "b5 + 自注意力模块",
    "社交池化 + 注意力加权",
    "全局平均池化 + 分类"
]

edges = [
    ("输入图像", "b1: 大卷积核提取特征"),
    ("b1: 大卷积核提取特征", "b2: 降维+局部特征提取"),
    ("b2: 降维+局部特征提取", "b3-b4: Inception多尺度融合"),
    ("b3-b4: Inception多尺度融合", "b5 + 自注意力模块"),
    ("b5 + 自注意力模块", "社交池化 + 注意力加权"),
    ("社交池化 + 注意力加权", "全局平均池化 + 分类")
]

# 添加节点
for node in nodes:
    G.add_node(node)

# 添加边
for edge in edges:
    G.add_edge(edge[0], edge[1])

# 设置布局
pos = nx.spring_layout(G, seed=42, k=2)

plt.figure(figsize=(12, 6))
plt.title("GoogLeNet + 自注意力 + 社交池化 流程图", fontsize=16)

# 绘制图形
nx.draw(G, pos,
        with_labels=True,
        node_size=3000,
        node_color='lightblue',
        font_size=12,
        font_weight='bold',
        edge_color='gray',
        arrows=True,
        arrowstyle='->',
        arrowsize=20)

# 显示图
plt.tight_layout()
plt.show()

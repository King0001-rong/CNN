import torch
print(torch.cuda.is_available())        # True 表示可用
print(torch.__version__)                # 查看 PyTorch 版本
print(torch.version.cuda)               # 查看对应的 CUDA 版本
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
a = torch.randn(1000, 1000, device=device)
b = torch.randn(1000, 1000, device=device)
c = a @ b  # 矩阵乘法测试
print(c.norm())  # 如果能输出数值，说明 GPU 正常

import torch
a = torch.randn(4)
print(a)
print(a*0.5)
# alpha * b + a, 维度不够的地方自动扩容
print(a.data.add_(a, alpha=0.5))
print(a)
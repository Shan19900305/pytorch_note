import torch

b = 10
m = 32
k = 64

a = torch.randn((b, m, k))

d = a.as_strided((b, k, m), (m * k, 1, k), 0)

for i in range(b):
    for j in range(m):
        for z in range(k):
            if a[i, j, z] != d[i, z, j]:
                print(a[i, j, z], d[i, z, j])

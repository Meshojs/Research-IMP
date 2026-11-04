import torch

class Model(torch.nn.Module):
    def __init__(self, infeature, outfeature) -> None:
        super().__init__()
        self.infeature = infeature
        self.outfeature = outfeature
        self.network =torch.nn.Sequential(
           torch.nn.Linear(infeature, 8),
           torch.nn.SiLU(),
           torch.nn.Linear(8,outfeature),
        )
    def forward(self, x):
        return self.network(x)
model = Model(2 , 2)
x = torch.randn(1 ,2)
y = torch.randn(1 ,2)
c = torch.nn.MSELoss()

"""
    AdamW not Adam + L2
"""
λ = 0.1
b1 = 0.9
b2 = 0.999
lr = 0.001
e = 1e-7
params = list(model.parameters())
m = [torch.zeros_like(p) for p in params]
v = [torch.zeros_like(p) for p in params]
for t in range(1 , 101):
    for p in params:
        p.grad = None
    y_pred = model(x)
    loss = c(y_pred, y)
    loss.backward()
    with torch.no_grad():
        for j , p in enumerate(params):
            g = p.grad
            m[j] = b1 * m[j] + (1 - b1) * g
            v[j] = b2 * v[j] + (1 - b2) * g**2
            m_hat = m[j] / (1 - b1**t)
            v_hat = v[j] / (1 - b2**t)
            p -= lr * (m_hat / (torch.sqrt(v_hat) + e) + λ * p)
    if t % 10 == 0:
        print(f"Loss : {loss}")

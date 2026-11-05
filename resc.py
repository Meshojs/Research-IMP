import torch as t
from torch import nn

class Model(nn.Module):
    def __init__(self, fn1 , fn2):
        super().__init__()
        self.network_1 = nn.Sequential(
            nn.Linear(fn1, 8),
            nn.SiLU(),
            nn.Linear(8, fn1),
        )
        self.skip_path = nn.Linear(fn1 , fn2)
        self.network_2 = nn.Sequential(
            nn.Linear(fn2, 32),
            nn.SiLU(),
            nn.Linear(32, fn2),
        )

    def forward(self , x):
        # y = F(x) + x
        y = self.network_1(x) + self.skip_path(x)
        y = self.network_2(y) + y
        return y


x = t.arange(0, 10).unsqueeze(dim=0).float()
y = (t.arange(0, 10) * 2).unsqueeze(dim=0).float()

print(x.dtype, y.dtype)



model = Model(10, 10)
c = nn.MSELoss()
optim = t.optim.Adam(model.parameters(), lr=0.001)

for i in range(1000):
    optim.zero_grad()
    y_pred = model(x)
    loss = c(y_pred , y)

    if i % 10 == 0 :
        print(f"Loss : {loss} ")

    loss.backward()
    optim.step()

x = t.arange(0, 10).unsqueeze(dim=0).float()

with t.no_grad():
    y_hat = model(x)
    print(t.round(y_hat))

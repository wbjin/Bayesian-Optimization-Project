import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch

def f(x1, x2):
    return (-x2+47)*np.sin(np.sqrt(x2+x1/2+47))-x1*np.sin(np.sqrt(np.abs(x1-x2+47)))

x, y = torch.meshgrid(torch.linspace(0, 100, 10), torch.linspace(0, 100, 10), indexing="ij")
X = train_x = torch.cat((
    x.contiguous().view(x.numel(), 1),
    y.contiguous().view(y.numel(), 1)),
    dim=1
)
xResult = []
yResult = []
z = []
for item in X:
    x = item[0].item()
    y = item[1].item()
    xResult.append(item[0].item())
    yResult.append(item[1].item())
    z.append(f(x, y))

fig2 = go.Figure(data = go.Scatter3d(x = np.array(xResult), y = np.array(yResult), z = np.array(z), mode = "markers"))
fig2.show()

xResult = np.array(xResult)
yResult = np.array(yResult)
z = np.array(z)

csvName = "3dtest.csv"
df = pd.DataFrame({"param1": xResult, "param2": yResult, "output": z})
df.to_csv(csvName, index=False)
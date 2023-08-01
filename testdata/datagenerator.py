import numpy as np
import plotly.graph_objects as go
import csv
import pandas as pd

i = 0.1
xData = []
yData = []
counter = 0
while (i <= 5):
    for x in range(8):
        xData.append(i)
    yData.append(np.log(i))
    yData.append(np.log(i+i/16))
    yData.append(np.log(i+i/18))
    yData.append(np.log(i+i/21))
    yData.append(np.log(i)*0.9)
    yData.append(np.log(i+i/16)*1.1)
    yData.append(np.log(i+i/18)*0.85)
    yData.append(np.log(i+i/20)*1.17)
    
    i+=0.1
    
xData = np.array(xData)
yData = np.array(yData)


for i in range(len(xData)):
    if (i >= len(xData)):
        break
    elif (i%3 == 0):
        xData = np.delete(xData, i)
        yData = np.delete(yData, i)
    elif (i%5 == 0):
        xData = np.delete(xData, [i, i+3])
        yData = np.delete(yData, [i, i+3])
    elif (i%5 == 0):
        xData = np.delete(xData, i)
        yData = np.delete(yData, i)

    
def remove(start, end, xData, yData):
    x = start
    array = []
    while (x < end):
        array.append(x)
        x+=1
    
    xData = np.delete(xData, array)
    yData = np.delete(yData, array)
    
    return xData, yData

xData, yData = remove(20, 30, xData, yData)
xData, yData = remove(37, 42, xData, yData)
xData, yData = remove(57, 69, xData, yData)
xData, yData = remove(83, 91, xData, yData)
xData, yData = remove(132, 153, xData, yData)
xData, yData = remove(179, 190, xData, yData)

fig = go.Figure(go.Scatter(x = xData, y = yData, mode = "markers"))
fig.show()



csvName = "expdatanoise.csv"
df = pd.DataFrame({"x": xData, "y": yData})
df.to_csv(csvName, index=False)
# with open(csvName, 'w') as csvfile: 
#     # creating a csv writer object 
#     csvwriter = csv.writer(csvfile) 
        
#     # writing the fields 
#     csvwriter.writerow(["x", "y"]) 
#     csvwriter.writerow(rows)




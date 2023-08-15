import numpy as np
import plotly.graph_objects as go
import csv
import pandas as pd

i = -1.5
xData = []
yData = []
counter = 0

def f(x):
    return x**2+np.cos(x**2)+np.sin(x**3)

while (i <= 1.5):
    for x in range(8):
        xData.append(i)
    yData.append(f(i))
    yData.append(f(i+i/16))
    yData.append(f(i+i/18))
    yData.append(f(i+i/21))
    yData.append(f(i)*0.9)
    yData.append(f(i+i/16)*1.1)
    yData.append(f(i+i/18)*0.85)
    yData.append(f(i+i/20)*1.17)
    
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

xData, yData = remove(30, 35, xData, yData)
xData, yData = remove(50, 55, xData, yData)
xData, yData = remove(70, 80, xData, yData)
xData, yData = remove(100, 110, xData, yData)



csvName = "demodata.csv"
df = pd.DataFrame({"param1": xData, "output": yData})
df.to_csv(csvName, index=False)
# with open(csvName, 'w') as csvfile: 
#     # creating a csv writer object 
#     csvwriter = csv.writer(csvfile) 
        
#     # writing the fields 
#     csvwriter.writerow(["x", "y"]) 
#     csvwriter.writerow(rows)




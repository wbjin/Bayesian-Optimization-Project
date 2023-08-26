from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

import numpy as np
from pymoo.core.problem import Problem

import pandas as pd

from pymoo.config import Config
Config.warnings['not_compiled'] = False

paramLowerBound = [7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
paramUpperBound = [9, 1, 1.31, 1.31, 1.31, 1.31, 1.31, 1.31, 1.31, 1.31, 1.31, 1.31, 1.31, 1.31, 1.31, 1.31, 1.31, 1.31, 1.31, 1.31, 1.31, 1.31, 1.31, 1.31]

class MyProblem(Problem):
    def __init__(self):
        super().__init__(n_var=24,
                         n_obj=4,
                         xl=paramLowerBound,
                         xu=paramUpperBound)

    def _evaluate(self, x, out, *args, **kwargs):
        filename = 'testdata/multiparam.xlsx'
        df = pd.read_excel(filename)
        df = df.drop(index = 0)
        y1 = df["DT"].to_numpy()[:23].astype('float64')
        y2 = df["r"].to_numpy()[:23].astype('float64')
        y3 = df["R0"].to_numpy()[:23].astype('float64')
        y4 = df["TL"].to_numpy()[:23].astype('float64')
        out["F"] = np.column_stack([y1, y2, y3, y4])

def main():
    problem = MyProblem()

    algorithm = NSGA2(pop_size=23)

    res = minimize(problem,
                algorithm,
                ('n_gen', 100),
                seed=1,
                verbose=False)

    bestIndex = 0
    bestSum = 0
    for i in range(len(res.F)):
        item = res.F[i]
        sum = item[1] + item[2] + item[3]-item[0]
        if sum > bestSum:
            bestIndex = i
    print("Optimization results: -------------------------------")
    print("Params A to W")
    for i in range(len(res.X[bestIndex])):
        param = chr(ord("A") + i)
        print(param, ": ", round(res.X[bestIndex][i], 3))

    print("Output: ")
    print("DT: ", res.F[bestIndex][0])
    print("r: ", res.F[bestIndex][1])
    print("R0: ", res.F[bestIndex][2])
    print("TL: ", res.F[bestIndex][3])

def run_nsga_optimization():
    problem = MyProblem()

    algorithm = NSGA2(pop_size=23)

    res = minimize(problem,
                algorithm,
                ('n_gen', 100),
                seed=1,
                verbose=False)

    bestIndex = 0
    bestSum = 0
    for i in range(len(res.F)):
        item = res.F[i]
        sum = item[1] + item[2] + item[3]-item[0]
        if sum > bestSum:
            bestIndex = i 
    return res.X[bestIndex], res.F[bestIndex]
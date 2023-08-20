import gpytorch
import torch

import numpy as np

import pandas as pd

from scipy.stats import norm

import plotly.graph_objects as go

import math
class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, params, numTasks):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=numTasks
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=numTasks, rank=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
    
class Optimizer:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.numInput = self.X.size(0)
        self.numTasks = len(self.y[0])
        if self.X.dim() == 1:
            self.params = 1
        else:
            self.params = len(self.X[0])
        if len(self.X != 0):
            if self.params == 1:
                self.domain = [self.X[0].item(), self.X[-1].item()]
                self.range = self.domain[1]-self.domain[0]
            elif self.params == 2:
                self.domain = [[self.X[0][0].item(), self.X[-1][0].item()], [self.X[0][1].item(), self.X[-1][1].item()]]
                self.range1 = self.domain[0][1]-self.domain[0][0]
                self.range2 = self.domain[1][1]-self.domain[1][0]
    
    def run(self):
        self.trainGP()
        for i in range(5):
            self.modelSurrogate()
            self.evaluateNext()
        return self.plot()

    def setOptimizationTarget(self, target):
        if target == "Maximize":
            self.target = max
            self.targetIndex = np.argmax
        else:
            self.target = min
            self.targetIndex = np.argmin
    
    def setKernel(self, kernelInput):
        self.kernel = gpytorch.kernels.MaternKernel(nu=0.5, ard_nums = self.params)
        if kernelInput == "matern":
            self.kernel = gpytorch.kernels.MaternKernel(nu=0.5, ard_nums = self.params) 
        if kernelInput == "sek":
            self.kernel = gpytorch.kernels.RBFKernel(ard_nums = self.params)  

    def setAcquisition(self, acquisitionInput):
        self.acquisition = self.probabilityOfImprovement 
        if acquisitionInput == "pi":
            self.acquisition = self.probabilityOfImprovement 
        
    def trainGP(self, trainIter = 10):
        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=self.numTasks)
        self.model = MultitaskGPModel(self.X, self.y, self.likelihood, self.params, self.numTasks)

        self.model.train()
        self.likelihood.train()

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)

        for i in range(20):
            optimizer.zero_grad()
            output = self.model(self.X)
            loss = -mll(output, self.y)
            loss.backward()
            optimizer.step()

        self.model.eval()
        self.observedPred = (self.model(self.X))
    
    def modelSurrogate(self):
        self.prediction = self.likelihood(self.observedPred).mean.detach().numpy()

    def probabilityOfImprovement(self):
        if self.params == 1:
            exploreX = self.range*torch.rand(25)+self.domain[0]
            bestPoint = self.target(self.prediction)
            observedPred = (self.model(exploreX))
            exploreY = self.likelihood(observedPred).mean.detach().numpy()
            stdev = np.sqrt(observedPred.variance.detach().numpy())
            z = (exploreY - bestPoint)/stdev
            cdf = norm.cdf(z)
            if self.target == max:
                index = np.argmax(cdf)
            else:
                cdf = -cdf
                index = np.argmin(cdf)
                newX = exploreX[index]

            return newX.double(), torch.tensor([index])
        elif self.params == 2:
            exploreX1 = self.range1*torch.rand(int(5))+self.domain[0][0]
            exploreX2 = self.range2*torch.rand(int(5))+self.domain[1][0]
            exploreX = torch.cat((
                exploreX1.contiguous().view(exploreX1.numel(), 1),
                exploreX2.contiguous().view(exploreX2.numel(), 1)),
                dim=1
            )
            bestPoint = self.target(self.prediction)
            observedPred = (self.model(exploreX))
            exploreY = self.likelihood(observedPred).mean.detach().numpy()
            stdev = np.sqrt(observedPred.variance.detach().numpy())
            z = (exploreY - bestPoint)/stdev
            cdf = norm.cdf(z)
            if self.target == max:
                index = np.argmax(cdf)
            else:
                cdf = -cdf
                index = np.argmin(cdf)
            newX = exploreX[index]
            return newX, index
    def evaluateNext(self):
        if self.params == 1:
            newX, index = self.acquisition()
            self.X.index_add(0, index, newX)
        elif self.params == 2:
            newX, index = self.acquisition()
            print(newX.unsqueeze(0))
            self.X = torch.cat((self.X, newX.unsqueeze(0)), 0)
            print(self.X)
    def result(self):
        print(self.prediction)
        
def main():
    filename = 'testdata/multiparam.xlsx'
    df = pd.read_excel(filename)
    df = df.drop(index = 0)
    A = torch.tensor(df["A"].to_numpy()[:23])
    B = torch.tensor(df["B"].to_numpy()[:23])
    C = torch.tensor(df["C"].to_numpy()[:23])
    D = torch.tensor(df["d"].to_numpy()[:23])
    E = torch.tensor(df["E"].to_numpy()[:-4])
    F = torch.tensor(df["F"].to_numpy()[:-4])
    G = torch.tensor(df["G"].to_numpy()[:-4])
    H = torch.tensor(df["H"].to_numpy()[:-4])
    I = torch.tensor(df["I"].to_numpy()[:-4])
    J = torch.tensor(df["J"].to_numpy()[:-4])
    K = torch.tensor(df["K"].to_numpy()[:-4])
    L = torch.tensor(df["L"].to_numpy()[:-4])
    M = torch.tensor(df["M"].to_numpy()[:-4])
    N = torch.tensor(df["N"].to_numpy()[:-4].astype('float64'))
    O = torch.tensor(df["O"].to_numpy()[:-4])
    P = torch.tensor(df["P"].to_numpy()[:-4])
    Q = torch.tensor(df["Q"].to_numpy()[:-4])
    R = torch.tensor(df["R"].to_numpy()[:-4])
    S = torch.tensor(df["S"].to_numpy()[:-4])
    T = torch.tensor(df["T"].to_numpy()[:-4])
    U = torch.tensor(df["U"].to_numpy()[:-4])
    V = torch.tensor(df["V"].to_numpy()[:-4])
    W = torch.tensor(df["W"].to_numpy()[:-4])

    X = torch.stack([A, B, C], -1).float()
    y1 = torch.from_numpy(df["DT"].to_numpy()[:23].astype('float64'))
    y2 = torch.from_numpy(df["r"].to_numpy()[:23].astype('float64'))
    y3 = torch.from_numpy(df["R0"].to_numpy()[:23].astype('float64'))
    y4 = torch.from_numpy(df["TL"].to_numpy()[:23].astype('float64'))
    y = torch.stack([y1, y2, y3, y4], -1).float()

    optimizer = Optimizer(X, y)
    optimizer.setOptimizationTarget("Maximize")
    optimizer.setKernel("sek")
    optimizer.setAcquisition("pi")
    optimizer.trainGP()
    optimizer.modelSurrogate()
    optimizer.result()

if __name__ == "__main__":
    main()
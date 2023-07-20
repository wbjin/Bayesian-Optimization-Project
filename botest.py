import os
import torch
import plotly
import numpy as np
import plotly.graph_objects as go
from botorch import fit_gpytorch_model
from botorch.models import SingleTaskGP, ModelListGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.optim import optimize_acqf

def targetFunction(individuals):
	result = []
	for x in individuals:
		result.append(np.exp(-(x-2)**2) + np.exp(-(x - 6)**2/10)+1/(x**2+1))
	
	return torch.tensor(result)

def plot(x, y, title):
	data = go.Scatter(x = x, y = y)
	fig = go.Figure(data = data)
	fig.update_layout(title = title)
	fig.show()

def generateInitialData(n = 10):
	trainX = torch.rand(10, 1)
	exactObj = targetFunction(trainX).unsqueeze(-1)
	bestObservedValue = exactObj.max().item()
	return trainX, exactObj, bestObservedValue

def getNextPoint(initX, initY, bestObservedValue, bounds, numPoints = 1):
	singleModel = SingleTaskGP(initX, initY)
	mll = ExactMarginalLogLikelihood(singleModel.likelihood, singleModel)
	
	fit_gpytorch_model(mll)
	EI = qExpectedImprovement(model = singleModel, best_f = bestObservedValue)
	candidates, _ = optimize_acqf(acq_function=EI, bounds = bounds, q = numPoints, num_restarts = 200,
			       					raw_samples = 512) 
	return candidates	

def optimize(initX, initY, bestInitY, bounds, numRuns = 10):
	for i in range(numRuns):
		print("Iteration ", i)
		newCandidate = getNextPoint(initX, initY, bestInitY, bounds)
		newResults = targetFunction(newCandidate).unsqueeze(-1)

		print("New point: ", newCandidate)
		initX = torch.cat([initX, newCandidate])
		initY = torch.cat([initY, newResults])

		title  = "Iteration " + str(i)
		bestInitY = initY.max().item()	
		plot(initX, initY, title)
		print("Best point: ", bestInitY)
	
def main():
	x = np.linspace(-2.0, 10.0, 100)
	y = targetFunction(x)
	plot(x, y, "Target Function")

	#Random initial data set
	initX, initY, bestInitY = generateInitialData(20)
	bounds = torch.tensor([[0.], [10.]])

	optimize(initX, initY, bestInitY, bounds, 10)


if __name__ == "__main__":
	main()
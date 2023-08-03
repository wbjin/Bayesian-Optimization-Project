import torch
import gpytorch
import plotly.graph_objects as go
import numpy

class ExactGPModel(gpytorch.models.ExactGP):
	def __init__(self, train_x, train_y, likelihood):
		super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
		self.mean_module = gpytorch.means.ConstantMean()
		self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

	def forward(self, x):
		mean_x = self.mean_module(x)
		covar_x = self.covar_module(x)
		return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
	
class Regression:
	def __init__(self, xPoints, yPoints):
		self.xPoints = torch.tensor(xPoints)
		self.yPoints = torch.tensor(yPoints)
	def regress(self):
		likelihood = gpytorch.likelihoods.GaussianLikelihood()
		model = ExactGPModel(self.xPoints, self.yPoints, likelihood)
		model.train()
		likelihood.train()
		optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
		mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
		for i in range(10):
			optimizer.zero_grad()
			output = model(self.xPoints)
			loss = -mll(output, self.yPoints)
			loss.backward()
			optimizer.step()
		model.eval()
		observedPred = likelihood(model(self.xPoints))
		mean = observedPred.mean.detach().numpy()
		lower, upper = observedPred.confidence_region()
		lower = lower.detach().numpy()
		upper = upper.detach().numpy()
		std = (mean[0]-lower[0])/2
		lower = mean - std
		upper = mean + std		
		fig = go.Figure(data = go.Scatter(x = self.xPoints, y = mean))
		fig.add_trace(go.Scatter(x = self.xPoints, y = self.yPoints, mode = "markers"))
		fig.add_trace(go.Scatter(x = self.xPoints, y = lower))
		fig.add_trace(go.Scatter(x = self.xPoints, y = upper, fill = 'tonexty', fillcolor = 'rgba(250,0,0,0.1)'))
		return fig
		
	
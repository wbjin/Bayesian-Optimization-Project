import torch
import matplotlib.pyplot as plt
import gpytorch
import math
import plotly.graph_objects as go


class ExactGPModel(gpytorch.models.ExactGP):
	def __init__(self, train_x, train_y, likelihood):
		super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
		self.mean_module = gpytorch.means.ConstantMean()
		self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

	def forward(self, x):
		mean_x = self.mean_module(x)
		covar_x = self.covar_module(x)
		return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
	
def plot(x, y, title):
	plt.plot(x, y)
	plt.show()

	
def main():
	xAxis = torch.linspace(0, 1, 100)
	yPoints = torch.sin(xAxis * (2 * math.pi)) + torch.randn(xAxis.size()) * math.sqrt(0.04)

	likelihood = gpytorch.likelihoods.GaussianLikelihood()
	model = ExactGPModel(xAxis, yPoints, likelihood)
	model.train()
	likelihood.train()

	# Use the adam optimizer
	optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

	# "Loss" for GPs - the marginal log likelihood
	mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

	for i in range(10):
		# Zero gradients from previous iteration
		optimizer.zero_grad()
		# Output from model
		output = model(xAxis)
		# Calc loss and backprop gradients
		loss = -mll(output, yPoints)
		loss.backward()
		optimizer.step()	

	model.eval()
	mean = likelihood(model(xAxis)).mean.detach().numpy()

	data = []
	data.append(go.Scatter(x= xAxis, y= mean))
	figOpt = go.Figure(data=data)
	figOpt.add_trace(go.Scatter(x = xAxis, y = yPoints, mode='markers'))
	figOpt.show()
			
if __name__ == "__main__":
	main()
import math
import torch
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.contrib.forecast import ForecastingModel, Forecaster, eval_crps
from pyro.infer.reparam import LinearHMMReparam, StableReparam, SymmetricStableReparam
from pyro.ops.tensor_utils import periodic_repeat
from pyro.ops.stats import quantile
import matplotlib.pyplot as plt



class MM(ForecastingModel):
	def model(self, zero_data, covariates):
		duration = zero_data.size(-2)
		init_dist = dist.Normal(0,10).expand([1]).to_event(1)
		obs_dist = dist.Normal(0,10).expand([1]).to_event(1)
		obs_matrix = torch.tensor([[1.]])
		trans_dist = dist.Normal(0,10).expand([1]).to_event(1)
		trans_matrix = torch.tensor([[1.]])
		pre_dist = dist.GaussianHMM(init_dist,trans_matrix, trans_dist, obs_matrix, obs_dist, duration=duration)
		prediction = periodic_repeat(torch.zeros(1, 1), duration, dim=-1).unsqueeze(-1)
		self.predict(pre_dist, prediction)

def main():
	data = torch.load('data.pt').transpose(-1, -2)
	data = data[0]
	data = data[:, None]
	pyro.set_rng_seed(1)
	pyro.clear_param_store()
	covariates = torch.zeros(len(data), 0)	
	forecaster = Forecaster(MM(), data[:700], covariates[:700], learning_rate=0.001)
	for name, value in forecaster.guide.median().items():
    		if value.numel() == 1:
        		print("{} = {:0.4g}".format(name, value.item()))

	samples = forecaster(data[:700], covariates, num_samples=100)
	p10, p50, p90 = quantile(samples, (0.1, 0.5, 0.9)).squeeze(-1)
	crps = eval_crps(samples, data[700:])
	plt.figure(figsize=(9, 3))
	plt.plot(torch.arange(700, 1404), p50, 'r-', label='forecast')
	plt.plot(torch.arange(700, 1404), data[700 : 1404], 'k-', label='truth')
	plt.title("Total hourly ridership (CRPS = {:0.3g})".format(crps))
	plt.ylabel("log(# rides)")
	plt.xlabel("Hour after 2011-01-01")
	plt.xlim(700, 1404)
	plt.legend(loc="best");

if __name__ == '__main__':
	main()		

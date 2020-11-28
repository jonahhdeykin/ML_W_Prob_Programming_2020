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
		duration = zero_data.size(1)
		init_dist = dist.Normal(0,10).expand([1]).to_event(1)
		obs_dist = dist.Normal(0,10).expand([1]).to_event(1)
		obs_matrix = torch.tensor([[1.]])
		trans_dist = dist.Normal(0,10).expand([1]).to_event(1)
		trans_matrix = torch.tensor([[1.]])
		pre_dist = dist.GaussianHMM(init_dist,trans_matrix, trans_dist, obs_matrix, obs_dist, duration=duration)
		prediction = pyro.sample("prediction", dist.Normal(0, 10).expand([1]))
		self.predict(pre_dist, prediction)
	



def main():
	data = torch.load('data.pt').transpose(-1, -2)
	data = data[0]
	data = data[:, None]
	pyro.set_rng_seed(1)
	pyro.clear_param_store()
	covariates = torch.zeros(len(data), 0)	
	forecaster = Forecaster(MM(), data[:700], covariates[:700], learning_rate=0.1)
	#for name, value in forecaster.guide.median().items():
    	#	if value.numel() == 1:
        #		print("{} = {:0.4g}".format(name, value.item()))

if __name__ == '__main__':
	main()		

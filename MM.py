import math
import torch
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.contrib.examples.bart import load_bart_od
from pyro.contrib.forecast import ForecastingModel, Forecaster, eval_crps
from pyro.infer.reparam import LinearHMMReparam, StableReparam, SymmetricStableReparam
from pyro.ops.tensor_utils import periodic_repeat
from pyro.ops.stats import quantile
import matplotlib.pyplot as plt


Class HMModel(ForecastingModel):
	def model(self, zero_data, covariates):
		duration = zero_data.size(-1)
		prediction = periodic_repeat(means, duration, dim=-1).unsqueeze(-1)
		init_dist = dist.Normal(0, 10).expand([1]).to_event(1)
		timescale = pyro.sample("timescale", dist.LogNormal(math.log(24), 1))
		trans_matrix = torch.exp(-1 / timescale)[..., None, None]
        	trans_scale = pyro.sample("trans_scale", dist.LogNormal(-0.5 * math.log(24), 1))
	        trans_dist = dist.Normal(0, trans_scale.unsqueeze(-1)).to_event(1)

		obs_matrix = torch.tensor([[1.]])
        	obs_scale = pyro.sample("obs_scale", dist.LogNormal(-2, 1))
        	obs_dist = dist.Normal(0, obs_scale.unsqueeze(-1)).to_event(1)

		noise_dist = dist.GaussianHMM(init_dist, trans_matrix, trans_dist, obs_matrix, obs_dist, duration=duration)
		self.predict(noise_dist, prediction)

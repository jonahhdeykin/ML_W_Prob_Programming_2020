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
		init_dist = dist.Normal(0,10).to_event(1)
		obs_dist = dist.Normal(0,10).to_event(1)
		obs_matrix = torch.tensor([[1.]])
		trans_dist = dist.Normal(0,10).to_event(1)
		trans_matrix = torch.tensor([[1.]])
		pre_dist = dist.GaussianHMM(init_dist,trans_matrix, trans_dist, obs_matrix, obs_dist, duration) 		



import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
from torch.distributions import constraints, transform_to

import pyro
import pyro.contrib.gp as gp

import math
import statistics as st
from time import sleep
import pickle
import read_data
import time
import numpy as np

torch.autograd.set_detect_anomaly(True)


class DDBL(nn.Module):
		def __init__(self, dims):
			super(DDBL, self).__init__()

			self.weights = nn.ModuleList()
			self.n_layers = len(dims)
			for i in range(0, len(dims) - 1):
				self.weights.append(nn.Linear(dims[i], dims[i+1]))

			self.weights.append(nn.Linear(dims[-1], 1))
			


		def forward(self, x):
			x = torch.transpose(x, 0, 1)
			for i in range(0, self.n_layers-1):
				x = torch.relu(self.weights[i](x))
			x = self.weights[-1](x)

			return x


		def get_params(self):
			return(list(self.parameters()))

		def update_params(self, difs):
			i = 0
			for f in self.parameters():
				f.data.sub_(difs[i])
				i += 1
		



class DTSBN():

	def __init__(self, dims, n_layers, num_steps_back):
		

		self.weights_dict = dict()

		for i in range(0, n_layers):
			self.weights_dict['w_{}'.format(3*i + 1)] = torch.normal(0, 0.01, size=(dims[i], dims[i], num_steps_back))
			self.weights_dict['w_{}'.format(3*i + 2)] = torch.normal(0, 0.01, size=(dims[i+1], dims[i]))
			self.weights_dict['w_{}'.format(3*i + 3)] = torch.normal(0, 0.01, size=(dims[i], dims[i+1], num_steps_back))
			self.weights_dict['b_{}'.format(i + 1)] = torch.normal(0., 0.01, size=(dims[i],))

			self.weights_dict['u_{}'.format(3*i + 1)] = torch.normal(0, 0.01, size=(dims[i], dims[i], num_steps_back))
			self.weights_dict['u_{}'.format(3*i + 2)] = torch.normal(0, 0.01, size=(dims[i], dims[i+1]))
			self.weights_dict['u_{}'.format(3*i + 3)] = torch.normal(0, 0.01, size=(dims[i], dims[i+1], num_steps_back))
			self.weights_dict['c_{}'.format(i + 1)] = torch.normal(0., 0.01, size=(dims[i],))


		self.weights_dict['w_{}'.format(3*n_layers + 1)] = torch.normal(0, 0.01, size=(dims[-1], dims[-1], num_steps_back))
		self.weights_dict['w_{}'.format(3*n_layers + 2)] = torch.normal(0, 0.01, size=(dims[-1], dims[-2]))
		self.weights_dict['w_{}'.format(3*n_layers + 3)] = torch.normal(0, 0.01, size=(dims[-1], dims[-1], num_steps_back))
		self.weights_dict['b_{}'.format(n_layers+1)] = torch.normal(0., 0.01, size=(dims[-1],))
		self.weights_dict['b_{}'.format(n_layers+2)] = torch.normal(0., 0.01, size=(dims[-1],))

		self.n_layers = n_layers
		self.DDBL = None
		self.dims = dims
		self.v = None
		self.c = None
		self.alpha = 0.8
		self.sig = nn.Sigmoid()
		self.nt = num_steps_back




	def sample_top_layer_gen(self, z_p, h_p):
		acc = torch.zeros((self.dims[0], ))
		for i in range(0, self.nt):
			acc = torch.add(acc, torch.add(torch.matmul(self.weights_dict['w_1'][:,:,i], z_p[:,-1-i]), 
				torch.matmul(self.weights_dict['w_3'][:,:,i], h_p[:,-1-i])))

		return(torch.bernoulli(self.sig(torch.add( acc, self.weights_dict['b_1']))))


	def sample_mid_layer_gen(self, z, h_p, v_p, index):
		acc = torch.zeros(self.dims[index])
		for i in range(0, self.nt):
			acc = torch.add(acc, torch.add(torch.matmul(self.weights_dict['w_{}'.format(3*index + 1)][:,:,i], h_p[:,-1-i]), 
				torch.matmul(self.weights_dict['w_{}'.format(3*(index + 1))][:,:,i], v_p[:,-1-i])))

		return (torch.bernoulli(self.sig(torch.matmul(self.weights_dict['w_{}'.format(3*index - 1)], z[:,-1]) + 
			acc + self.weights_dict['b_{}'.format(index+1)])))

	def sample_bottom_layer_gen(self, h, v_p,):
		acc_u = torch.zeros((self.dims[-1], ))
		acc_v = torch.zeros((self.dims[-1], ))

		for i in range(0, self.nt):
			acc_u = torch.add(acc_u, torch.matmul(self.weights_dict['w_{}'.format(3*self.n_layers + 1)][:,:,i], v_p[:,-1-i]))
			acc_v = torch.add(acc_v, torch.matmul(self.weights_dict['w_{}'.format(3*self.n_layers + 3)][:,:,i], v_p[:,-1-i]))

		return (torch.normal((acc_u + 
			torch.matmul(self.weights_dict['w_{}'.format(3*self.n_layers - 1)], h[:,-1]) + 
			self.weights_dict['b_{}'.format(self.n_layers+1)]), 
			torch.exp(acc_v + 
			torch.matmul(self.weights_dict['w_{}'.format(3*self.n_layers + 2)], h[:,-1]) + 
			self.weights_dict['b_{}'.format(self.n_layers+2)])))

	def sample_layer_rec(self, v_p, v, h_p, index):
		index = self.n_layers - index - 1 
		acc = torch.zeros((self.dims[index], ))
		
		for i in range(0, self.nt):
			acc = torch.add(acc, torch.add(torch.matmul(self.weights_dict['u_{}'.format(3*index + 1)][:,:,i], h_p[:,-1-i]), 
				torch.matmul(self.weights_dict['u_{}'.format(3*index + 3)][:,:,i], v_p[:,-1-i])))

		return (torch.bernoulli(self.sig(torch.add(torch.add(torch.matmul(self.weights_dict['u_{}'.format(3*index + 2)], v[:,-1]), acc), self.weights_dict['c_{}'.format(index+1)]))))


	def compute_grads(self, x_list, DDBL_net, criterion):
		with torch.no_grad():
			states = []


			for i in range(0, self.n_layers):
				states.append(torch.zeros(self.dims[i], x_list.size()[1]))

			states[-1][:, 0] = torch.bernoulli(self.sig(torch.add(torch.matmul(self.weights_dict['u_{}'.format(3*self.n_layers - 1)], x_list[:,0]), self.weights_dict['c_{}'.format(self.n_layers)])))

			for i in range(0, self.n_layers-1):
				states[ - 2 - i][:, 0] = torch.bernoulli(self.sig( torch.add(torch.matmul(self.weights_dict['u_{}'.format(3*self.n_layers - 1 - 3*(i+1))], states[ - 1 - i][:,0].clone()), self.weights_dict['c_{}'.format(self.n_layers - i - 1)])))



			for t in range(1, x_list.size()[1]):
				accumulator = []
				for i in range(0, self.n_layers):
					accumulator.append(torch.zeros((self.dims[i], )).detach())
				for d in range(0, min(t-1, self.nt)):
					for i in range(0, self.n_layers - 1):
						
						accumulator[i] = torch.add(accumulator[i].clone(), 
							torch.add(torch.matmul(self.weights_dict['u_{}'.format(3*i+1)][:,:,d], states[i][:,t-d-1].clone()), 
								torch.matmul(self.weights_dict['u_{}'.format(3*i+3)][:,:,d], states[i+1][:,t-d-1].clone())))
					accumulator[-1] = torch.add(accumulator[-1].clone(), 
						torch.add(torch.matmul(self.weights_dict['u_{}'.format(3*self.n_layers - 2)][:,:,d], states[-1][:,t-d-1].clone()), 
							torch.matmul(self.weights_dict['u_{}'.format(3*self.n_layers)][:,:,d], x_list[:,t-d-1])))

			
				states[-1][:, t] = torch.bernoulli(self.sig(torch.add(torch.matmul(self.weights_dict['u_{}'.format(3*self.n_layers - 1)], x_list[:,t]), 
					torch.add(self.weights_dict['c_{}'.format(self.n_layers)], accumulator[-1]))))

				for i in range(0, self.n_layers-1):
					states[ - 2 - i][:, t] = torch.bernoulli(self.sig( torch.add(torch.matmul(self.weights_dict['u_{}'.format(3*self.n_layers - 1 - 3*(i+1))], states[ - 1 - i][:,t].clone()), 
						torch.add(self.weights_dict['c_{}'.format(self.n_layers - i - 1)], accumulator[-2-i]))))



			phi_gen = []
			phi_rec = []
			for i in range(0, self.n_layers):
				phi_gen.append(torch.zeros((self.dims[i], x_list.size()[1])).detach())
				phi_rec.append(torch.zeros((self.dims[i], x_list.size()[1])).detach())

			mu = torch.zeros((self.dims[-1], x_list.size()[1])).detach()
			log_sig = torch.zeros((self.dims[-1], x_list.size()[1])).detach()

			phi_gen[0][:,0] = self.weights_dict['b_1']
			for i in range(0, self.n_layers - 1):
				phi_gen[i+1][:,0] = torch.add(torch.matmul(self.weights_dict['w_{}'.format(3*i+2)], states[i][:,0]),
					self.weights_dict['b_{}'.format(i+2)])

			for i in range(0, self.n_layers-1):
				phi_rec[i][:,0] = torch.add(torch.matmul(self.weights_dict['u_{}'.format(3*i+2)], states[i+1][:,0]),
					self.weights_dict['c_{}'.format(i+1)])

			phi_rec[-1][:,0] = torch.add(torch.matmul(self.weights_dict['u_{}'.format(3*self.n_layers-1)], x_list[:,0]),
					self.weights_dict['c_{}'.format(self.n_layers)])

			mu[:,0] = torch.add(torch.matmul(self.weights_dict['w_{}'.format(3*self.n_layers-1)], states[-1][:,0]), 
				self.weights_dict['b_{}'.format(self.n_layers+1)])
			
			log_sig[:,0] = torch.add(torch.matmul(self.weights_dict['w_{}'.format(3*self.n_layers+2)], states[-1][:,0]), 
				self.weights_dict['b_{}'.format(self.n_layers+2)])


			for t in range(1, x_list.size()[1]):
				accumulator_b = []
				for i in range(0, self.n_layers+1):
					accumulator_b.append(torch.zeros((self.dims[i], )).detach())

				accumulator_b.append(torch.zeros((self.dims[-1], )).detach())
				accumulator_c = []
				for i in range(0, self.n_layers):
					accumulator_c.append(torch.zeros((self.dims[i], )).detach())

				for d in range(0, min(t-1, self.nt)):
					
					for i in range(0, self.n_layers-1):
						accumulator_b[i] = torch.add(accumulator_b[i], 
							torch.add(torch.matmul(self.weights_dict['w_{}'.format(3*i+1)][:,:,d], states[i][:,t-d-1]), 
								torch.matmul(self.weights_dict['w_{}'.format(3*i+3)][:,:,d], states[i+1][:,t-d-1])))

						accumulator_c[i] = torch.add(accumulator_c[i], 
							torch.add(torch.matmul(self.weights_dict['u_{}'.format(3*i+1)][:,:,d], states[i][:,t-d-1]), 
								torch.matmul(self.weights_dict['u_{}'.format(3*i+3)][:,:,d], states[i+1][:,t-d-1])))
					
			
					accumulator_b[-3] = torch.add(accumulator_b[-3], 
						torch.add(torch.matmul(self.weights_dict['w_{}'.format(3*self.n_layers - 2)][:,:,d], states[-1][:,t-d-1]), 
							torch.matmul(self.weights_dict['w_{}'.format(3*self.n_layers)][:,:,d], x_list[:,t-d-1])))

					accumulator_b[-2] = torch.add(accumulator_b[-2], 
						torch.matmul(self.weights_dict['w_{}'.format(3*self.n_layers+1)][:,:,d], x_list[:, t-d-1]))

					accumulator_b[-1] = torch.add(accumulator_b[-1], 
						torch.matmul(self.weights_dict['w_{}'.format(3*self.n_layers+3)][:,:,d], x_list[:, t-d-1]))

					accumulator_c[-1] = torch.add(accumulator_c[-1], 
						torch.add(torch.matmul(self.weights_dict['u_{}'.format(3*self.n_layers - 2)][:,:,d], states[-1][:,t-d-1]), 
							torch.matmul(self.weights_dict['u_{}'.format(3*self.n_layers)][:,:,d], x_list[:,t-d-1])))

				phi_gen[0][:,t] = torch.add(self.weights_dict['b_1'], accumulator_b[0])

				for i in range(0, self.n_layers - 1):
					phi_gen[i+1][:,t] = torch.add(torch.add(torch.matmul(self.weights_dict['w_{}'.format(3*i+2)], states[i][:,t]),
						self.weights_dict['b_{}'.format(i+2)]), accumulator_b[i+1])

				for i in range(0, self.n_layers-1):
					phi_rec[i][:,t] = torch.add(torch.add(torch.matmul(self.weights_dict['u_{}'.format(3*i+2)], states[i+1][:,t]),
						self.weights_dict['c_{}'.format(i+1)]), accumulator_c[i])

				phi_rec[-1][:,t] = torch.add(torch.add(torch.matmul(self.weights_dict['u_{}'.format(3*self.n_layers-1)], x_list[:,t]),
						self.weights_dict['c_{}'.format(self.n_layers)]), accumulator_c[-1])

				mu[:,t] = torch.add(torch.add(torch.matmul(self.weights_dict['w_{}'.format(3*self.n_layers-1)], states[-1][:,t]), 
					self.weights_dict['b_{}'.format(self.n_layers+1)]), accumulator_b[-2])

				log_sig[:,t] = torch.add(torch.add(torch.matmul(self.weights_dict['w_{}'.format(3*self.n_layers+2)], states[-1][:,t]), 
					self.weights_dict['b_{}'.format(self.n_layers+2)]), accumulator_b[-1])


			lpr = torch.zeros((x_list.size()[1],)).detach()
			for i in range(0, self.n_layers):
				
				lpr += torch.sum(torch.sub(torch.mul(phi_gen[i], states[i]), torch.log(torch.add(torch.exp(phi_gen[i]), 1))), 0)
				
			logl = -torch.sum(torch.add(torch.add(log_sig, torch.div(torch.square(torch.sub(x_list, mu)), torch.mul(torch.exp(torch.mul(log_sig, 2)), 2))), 0.5*math.log(2*math.pi)), 0)

			lps = torch.zeros((x_list.size()[1],)).detach()
			for i in range(0, self.n_layers):
				lps += torch.sum(torch.sub(torch.mul(phi_rec[i], states[i]), torch.log(torch.add(torch.exp(phi_rec[i]), 1))), 0)


			
			ll = torch.add(lpr, torch.sub(logl, lps)).detach()
			o_ll = torch.zeros((1, x_list.size()[1])).detach()
			o_ll[0] = torch.clone(ll)
			lb = torch.mean(ll)


		preds = DDBL_net(x_list)
		loss = criterion(torch.transpose(preds, 0, 1), o_ll)
		DDBL_net.zero_grad()
		loss.backward()

		
		with torch.no_grad():
			net_derivs = []
			for f in DDBL_net.parameters():
				 net_derivs.append(f.grad.data)

			
			loss = loss.item()
			ll = torch.sub(ll, torch.transpose(preds, 0, 1))

			if self.c is not None:
				self.c = self.c*self.alpha + (1-self.alpha)*torch.mean(ll)
				self.v = self.v*self.alpha + (1-self.alpha)*torch.var(ll)
			else:
				self.c = torch.mean(ll)
				self.v = torch.var(ll)

			ll = torch.div(torch.sub(ll, self.c), max(1, math.sqrt(self.v.item())))
		
			derivs_dict = dict()
			for w in self.weights_dict:
				derivs_dict[w] = torch.zeros(self.weights_dict[w].size())




			chi_gen = torch.sub(states[0], self.sig(phi_gen[0]))
			if len(states) > 1:
				for d in range(0, self.nt):
					derivs_dict['w_1'][:,:,d] = torch.div(torch.matmul(chi_gen[:,d+1:x_list.size()[1]], 
						torch.transpose(states[0][:,0:x_list.size()[1]-d-1], 0, 1)), x_list.size()[1]-d)
					derivs_dict['w_3'][:,:,d] = torch.div(torch.matmul(chi_gen[:,d+1:x_list.size()[1]], 
						torch.transpose(states[1][:,0:x_list.size()[1]-d-1], 0, 1)), x_list.size()[1]-d)
				derivs_dict['b_1'] = torch.div(torch.sum(chi_gen, 1), x_list.size()[1])
			


			for i in range(0, self.n_layers-2):
				chi_gen = torch.sub(states[i+1], self.sig(phi_gen[i+1]))
				derivs_dict['w_{}'.format(3*i+2)] = torch.div(torch.matmul(chi_gen, torch.transpose(states[i], 0, 1)), x_list.size()[1])
				for d in range(0, self.nt):
					derivs_dict['w_{}'.format(3*i+4)][:,:,d] = torch.div(torch.matmul(chi_gen[:,d+1:x_list.size()[1]], 
						torch.transpose(states[i+1][:,0:x_list.size()[1]-d-1], 0, 1)), x_list.size()[1]-d-1)
					derivs_dict['w_{}'.format(3*i+6)][:,:,d] = torch.div(torch.matmul(chi_gen[:,d+1:x_list.size()[1]], 
						torch.transpose(states[i+2][:,0:x_list.size()[1]-d-1], 0, 1)), x_list.size()[1]-d-1)
				derivs_dict['b_{}'.format(i+2)] = torch.div(torch.sum(chi_gen, 1), x_list.size()[1])

			chi_gen = torch.sub(states[-1], self.sig(phi_gen[-1]))
			if len(states) > 1:
				derivs_dict['w_{}'.format(3*self.n_layers-4)]= torch.div(torch.matmul(chi_gen, torch.transpose(states[-2], 0, 1)), x_list.size()[1])

			for d in range(0, self.nt):
				derivs_dict['w_{}'.format(3*self.n_layers-2)][:,:,d] = torch.div(torch.matmul(chi_gen[:,d+1:x_list.size()[1]], 
					torch.transpose(states[-1][:,0:x_list.size()[1]-d-1], 0, 1)), x_list.size()[1]-d-1)
				derivs_dict['w_{}'.format(3*self.n_layers)][:,:,d] = torch.div(torch.matmul(chi_gen[:,d+1:x_list.size()[1]], 
					torch.transpose(x_list[:,0:x_list.size()[1]-d-1], 0, 1)), x_list.size()[1]-d-1)
			derivs_dict['b_{}'.format(self.n_layers)] = torch.div(torch.sum(chi_gen, 1), x_list.size()[1])


			chi_mu = torch.div(torch.sub(x_list, mu), torch.exp(torch.mul(log_sig, 2)))
			chi_sig = torch.sub(torch.div(torch.square(torch.sub(x_list, mu)), torch.exp(torch.mul(log_sig, 2))), 1)
			derivs_dict['w_{}'.format(3*self.n_layers-1)] = torch.div(torch.matmul(chi_mu, torch.transpose(states[-1], 0, 1)), x_list.size()[1])
			derivs_dict['w_{}'.format(3*self.n_layers+2)] = torch.div(torch.matmul(chi_sig, torch.transpose(states[-1], 0, 1)), x_list.size()[1])
			for d in range(0, self.nt):
				derivs_dict['w_{}'.format(3*self.n_layers+1)][:,:,d] = torch.div(torch.matmul(chi_mu[:,d+1:x_list.size()[1]], 
					torch.transpose(x_list[:,0:x_list.size()[1]-d-1], 0, 1)), x_list.size()[1]-d-1)
				derivs_dict['w_{}'.format(3*self.n_layers+3)][:,:,d] = torch.div(torch.matmul(chi_sig[:,d+1:x_list.size()[1]], 
					torch.transpose(x_list[:,0:x_list.size()[1]-d-1], 0, 1)), x_list.size()[1]-d-1)
			derivs_dict['b_{}'.format(self.n_layers+1)] = torch.div(torch.sum(chi_mu, 1), x_list.size()[1])
			derivs_dict['b_{}'.format(self.n_layers+2)] = torch.div(torch.sum(chi_sig, 1), x_list.size()[1])

			ll_temp = torch.zeros((1, x_list.size()[1]))

			ll_temp[0] = ll

			for i in range(0, self.n_layers-1):

				chi_rec = torch.mul(torch.sub(states[i], self.sig(phi_rec[i])), ll_temp)
				derivs_dict['u_{}'.format(3*i+2)] = torch.div(torch.matmul(chi_rec, torch.transpose(states[i+1], 0, 1)), x_list.size()[1])
				for d in range(0, self.nt):
					derivs_dict['u_{}'.format(3*i+1)][:,:,d] = torch.div(torch.matmul(chi_rec[:,d+1:x_list.size()[1]], 
						torch.transpose(states[i][:,0:x_list.size()[1]-d-1], 0, 1)), x_list.size()[1]-d-1)
					derivs_dict['u_{}'.format(3*i+3)][:,:,d] = torch.div(torch.matmul(chi_rec[:,d+1:x_list.size()[1]], 
						torch.transpose(states[i+1][:,0:x_list.size()[1]-d-1], 0, 1)), x_list.size()[1]-d-1)
				derivs_dict['c_{}'.format(i+1)] = torch.div(torch.sum(chi_rec, 1), x_list.size()[1])

			chi_rec = torch.mul(torch.sub(states[-1], self.sig(phi_rec[-1])), ll)
			derivs_dict['u_{}'.format(3*self.n_layers-1)] = torch.div(torch.matmul(chi_rec, torch.transpose(x_list, 0, 1)), x_list.size()[1])
			for d in range(0, self.nt):
				derivs_dict['u_{}'.format(3*self.n_layers-2)][:,:,d] = torch.div(torch.matmul(chi_rec[:,d+1:x_list.size()[1]], 
					torch.transpose(states[-1][:,0:x_list.size()[1]-d-1], 0, 1)), x_list.size()[1]-d-1)
				derivs_dict['u_{}'.format(3*self.n_layers)][:,:,d] = torch.div(torch.matmul(chi_rec[:,d+1:x_list.size()[1]], 
					torch.transpose(x_list[:,0:x_list.size()[1]-d-1], 0, 1)), x_list.size()[1]-d-1)
			derivs_dict['c_{}'.format(self.n_layers)] = torch.div(torch.sum(chi_rec, 1), x_list.size()[1])

			return derivs_dict, net_derivs, lb.item(), loss 
		



	def update_params(self, difs):
		for key in difs:
			self.weights_dict[key] = torch.add(self.weights_dict[key], difs[key])

	def save_model(self, outpath):
		with open(outpath, 'wb') as output:
			pickle.dump(self.weights_dict, output)


	def load_weights(self, inpath):
		with open(inpath, 'rb') as pkl_file:
			self.weights_dict = pickle.load(pkl_file)




	

def Adam(DTSBN, DDBL, data, criterion, batch_size = 50, epochs=75, alpha = 0.0001, beta_1 = 0.9, beta_2 = 0.999, eta = 1e-8, t_max = None, Noisy = True):
	ends = [(i*50, i*50+batch_size) for i in range(0, int(data.size()[1]/batch_size))]
	ends.append((int(data.size()[1]/batch_size)*50, data.size()[1]))
	first_pass = True
	t = 0
	epoch_losses = []

	if t_max is None:
		for e in range(0, epochs): 
			e_loss = 0
			for j in range(0, len(ends)):
				t += 1
				
				DTSBN_derivs, DDBL_derivs, c, loss = DTSBN.compute_grads(data[:,ends[j][0]:ends[j][1]], DDBL, criterion)
				e_loss += c * (ends[j][1] - ends[j][0])

				for key in DDBL_derivs:
					DTSBN_derivs[key] = torch.mul(DTSBN_derivs[key], (ends[j][1] - ends[j][0] )/batch_size)

				for i in range(0, len(DDBL_derivs)):
					DDBL_derivs[i] = torch.mul(DDBL_derivs[i], (ends[j][1] - ends[j][0] )/batch_size)

				if Noisy:
					print('epoch {}/{}, example {}/{} DTSBN Loss: {}, DDBL Loss: {}'.format(e+1, epochs, j+1, len(ends), c.item(), loss))
				with torch.no_grad():
					if first_pass:
						m_DTSBN = dict()
						v_DTSBN = dict()

						m_DDBL = []
						v_DDBL = []

						for key in DTSBN_derivs:
							m_DTSBN[key] = torch.zeros(DTSBN_derivs[key].size())
							v_DTSBN[key] = torch.zeros(DTSBN_derivs[key].size())

						for deriv in DDBL_derivs:
							m_DDBL.append(torch.zeros(deriv.size()))
							v_DDBL.append(torch.zeros(deriv.size()))

						first_pass = False


					for key in m_DTSBN:
						m_DTSBN[key] = torch.add(torch.mul(m_DTSBN[key], beta_1), torch.mul(DTSBN_derivs[key], (1-beta_1)))
						v_DTSBN[key] = torch.add(torch.mul(v_DTSBN[key], beta_2), torch.mul(torch.square(DTSBN_derivs[key]), (1-beta_2)))

					for i in range(0, len(DDBL_derivs)):
						m_DDBL[i] = torch.add(torch.mul(m_DDBL[i], beta_1), torch.mul(DDBL_derivs[i], (1-beta_1)))
						v_DDBL[i] = torch.add(torch.mul(v_DDBL[i], beta_2), torch.mul(torch.square(DDBL_derivs[i]), (1-beta_2)))
						


					alpha_t = alpha*math.sqrt(1-math.pow(beta_2, t))/(1-math.pow(beta_1, t))

					difs_DTSBN = dict()
					difs_DDBL = []

					for key in m_DTSBN:
						difs_DTSBN[key] = torch.mul(torch.div(m_DTSBN[key], torch.add(torch.sqrt(torch.abs(v_DTSBN[key].detach())), eta)), alpha_t)
					
					for i in range(0, len(DDBL_derivs)):
						difs_DDBL.append(torch.mul(torch.div(m_DDBL[i], torch.add(torch.sqrt(torch.abs(v_DDBL[i].detach())), eta)), alpha_t))

					DTSBN.update_params(difs_DTSBN)
					DDBL.update_params(difs_DDBL)

			e_loss = e_loss/data.size()[1]
			print("epoch {} average loss: {}".format(e+1, e_loss))
			epoch_losses.append(e_loss)
		return epoch_losses
	else:
		epoch_losses_past = torch.zeros((len(ends),))
		epoch_losses_current = torch.zeros((len(ends),))
		start = time.time()
		e = 0
		epoch_loss = 0
		while time.time() - start < t_max:
			j = 0
			while j < len(ends) and time.time() - start < t_max:
				

				t += 1
				DTSBN_derivs, DDBL_derivs, c, loss = DTSBN.compute_grads(data[:,ends[j][0]:ends[j][1]], DDBL, criterion)
				epoch_losses_current[j] = c * (ends[j][1] - ends[j][0] )
				for key in DTSBN_derivs:
					DTSBN_derivs[key] = torch.mul(DTSBN_derivs[key], (ends[j][1] - ends[j][0] )/batch_size)

				for i in range(0, len(DDBL_derivs)):
					DDBL_derivs[i] = torch.mul(DDBL_derivs[i], (ends[j][1] - ends[j][0] )/batch_size)

				if Noisy:
					print('epoch {}/{}, example {}/{} DTSBN Loss: {}, DDBL Loss: {}'.format(e+1, epochs, j+1, len(ends), c, loss))
				with torch.no_grad():
					if first_pass:
						m_DTSBN = dict()
						v_DTSBN = dict()

						m_DDBL = []
						v_DDBL = []

						for key in DTSBN_derivs:
							m_DTSBN[key] = torch.zeros(DTSBN_derivs[key].size())
							v_DTSBN[key] = torch.zeros(DTSBN_derivs[key].size())

						for deriv in DDBL_derivs:
							m_DDBL.append(torch.zeros(deriv.size()))
							v_DDBL.append(torch.zeros(deriv.size()))

						first_pass = False


					for key in m_DTSBN:
						m_DTSBN[key] = torch.add(torch.mul(m_DTSBN[key], beta_1), torch.mul(DTSBN_derivs[key], (1-beta_1)))
						v_DTSBN[key] = torch.add(torch.mul(v_DTSBN[key], beta_2), torch.mul(torch.square(DTSBN_derivs[key]), (1-beta_2)))

					for i in range(0, len(DDBL_derivs)):
						m_DDBL[i] = torch.add(torch.mul(m_DDBL[i], beta_1), torch.mul(DDBL_derivs[i], (1-beta_1)))
						v_DDBL[i] = torch.add(torch.mul(v_DDBL[i], beta_2), torch.mul(torch.square(DDBL_derivs[i]), (1-beta_2)))
						


					alpha_t = alpha*math.sqrt(1-math.pow(beta_2, t))/(1-math.pow(beta_1, t))

					difs_DTSBN = dict()
					difs_DDBL = []

					for key in m_DTSBN:
						difs_DTSBN[key] = torch.mul(torch.div(m_DTSBN[key], torch.add(torch.sqrt(torch.abs(v_DTSBN[key])), eta)), alpha_t)
					
					for i in range(0, len(DDBL_derivs)):
						difs_DDBL.append(torch.mul(torch.div(m_DDBL[i], torch.add(torch.sqrt(torch.abs(v_DDBL[i].detach())), eta)), alpha_t))

					DTSBN.update_params(difs_DTSBN)
					DDBL.update_params(difs_DDBL)

				if e == 0:
					epoch_loss = torch.sum(epoch_losses_current).item()/ends[j][1]

				else:
					if ends[j][1] != data.size()[1]:
						epoch_loss = (torch.sum(epoch_losses_current) + torch.sum(epoch_losses_past[j:])).item()/data.size()[1]
					
					else:
						epoch_loss = torch.sum(epoch_losses_current).item()/data.size()[1]

				j +=1
			
			epoch_losses_past = epoch_losses_current.clone()
			epoch_losses_current = torch.zeros((len(ends),))

			e += 1
			
			print("epoch {} average loss: {}".format(e, epoch_loss))
			
		return epoch_loss

def hyperband(budget, ranges, eta, data):
	
	s_max = math.floor(math.log(budget))
	b = (s_max+1)*budget

	s_list = [i for i in range(0, s_max + 1)]
	s_list.reverse()
	best_configs = torch.zeros((s_max+1, 5))
	
	tot = 0
	for s in s_list:
		n =  math.ceil((b/budget)*(eta**s)/(s+1))
		r = int(budget*eta**-s)
		configs = []
		for _ in range(0, n):
			rands = np.random.uniform(size=(n,))
			config = [((ranges[i][1]-ranges[i][0])*rands[i])+ranges[i][0] for i in range(0, 3)]

			
			dims = [data.size()[0]]
			for _ in range(0, int(config[1])):
				dims.append(max(int(dims[-1] / config[0]), 1))

			
			DDBL_net = DDBL(dims)
			dims.reverse()
			DTSBN_net = DTSBN(dims, int(config[1]), int(config[2]))

			configs.append([0, (DTSBN_net, DDBL_net), config])



		for i in range(0, s+1):
			n_i = math.floor(n*eta**-i)
			r_i = r * eta**i
			new_configs = []
			for config in configs:
				tot += r_i

				loss = Adam(config[1][0], config[1][1], data, nn.MSELoss(),  t_max=r_i*60, Noisy = False)
				new_configs.append([loss, config[1], config[2]])

			new_configs.sort(key=lambda x: -x[0])

			
			configs = new_configs[:max(1, math.floor(n_i/eta))]

			print(i, s, s_max)
			if len(configs) == 1:
				break

		best_configs[s_max-s][0] = configs[0][0]
		best_configs[s_max-s][1] = configs[0][2][0]
		best_configs[s_max-s][2] = configs[0][2][1]
		best_configs[s_max-s][3] = configs[0][2][2]


	return(best_configs)


class bayesian_op():
	def __init__(self):
		self.ranges = None
		self.data = None
		self.gpmodel = None
		self.time_limit = None

		



	def run(self, ranges, data, warmup, bayesian, time_limit):

		self.ranges = ranges
		self.data = data
		self.time_limit = time_limit

		
		X = torch.rand((warmup, 3))
		y = torch.zeros((warmup,))

		for j in range(0, warmup):
			configs = [((self.ranges[i][1]-self.ranges[i][0])*X[j][i].item())+self.ranges[i][0] for i in range(0, 3)]
			dims = [self.data.size()[0]]
			

			for _ in range(0, int(configs[1])):
				dims.append(max(1, int(dims[-1] / configs[0])))

			DDBL_net = DDBL(dims)
			dims.reverse()
			DTSBN_net = DTSBN(dims, int(configs[1]), int(configs[2]))

			y[j] = Adam(DTSBN_net, DDBL_net, self.data, nn.MSELoss(), t_max = self.time_limit, Noisy = False)
			print('warmup step {}/{}'.format(j+1, warmup))
		self.gpmodel = gp.models.GPRegression(X, y, gp.kernels.Matern52(input_dim=3),
								 noise=torch.tensor(0.1), jitter=1.0e-4)


		for j in range(0, bayesian):
			x_min = self.next_x()
			self.update_posterior(x_min)
			print('bayesian step {}/{}'.format(j+1, bayesian))

		return (self.gpmodel.X, self.gpmodel.y)



	def update_posterior(self, x_new):
		
		

		configs = [((self.ranges[i][1]-self.ranges[i][0])*x_new[i].item())+self.ranges[i][0] for i in range(0, 3)]
		dims = [self.data.size()[0]]

		for _ in range(0, int(configs[1])):
			dims.append(max(1, int(dims[-1] / configs[0])))

		DDBL_net = DDBL(dims)
		dims.reverse()
		DTSBN_net = DTSBN(dims, int(configs[1]), int(configs[2]))
		

		
		y = torch.tensor(Adam(DTSBN_net, DDBL_net, self.data, nn.MSELoss(),  t_max = self.time_limit, Noisy = False))
		x_fix = torch.zeros((1, 3))
		x_fix[0] = x_new
		X = torch.cat([self.gpmodel.X, x_fix])
		y_fix = torch.zeros((1,))
		y_fix[0] = y
		y = torch.cat([self.gpmodel.y, y_fix])

		self.gpmodel.set_data(X, y)

		# optimize the GP hyperparameters using Adam with lr=0.001
		optimizer = torch.optim.Adam(self.gpmodel.parameters(), lr=0.001)
		gp.util.train(self.gpmodel, optimizer)

	def lower_confidence_bound(self, x, kappa=2):

		x_fix = torch.zeros((1, x.size()[0]))
		x_fix[0] = x
		mu, variance = self.gpmodel(x_fix, full_cov=False, noiseless=False)
		sigma = variance.sqrt()
		return -mu - kappa * sigma


	def find_a_candidate(self, x_init, lower_bound=0, upper_bound=1):
		# transform x to an unconstrained domain
		constraint = constraints.interval(lower_bound, upper_bound)
		unconstrained_x_init = transform_to(constraint).inv(x_init)
		unconstrained_x = unconstrained_x_init.clone().detach().requires_grad_(True)
		minimizer = optim.LBFGS([unconstrained_x], line_search_fn='strong_wolfe')

		def closure():
			minimizer.zero_grad()
			x = transform_to(constraint)(unconstrained_x)
			y = self.lower_confidence_bound(x)
			autograd.backward(unconstrained_x, autograd.grad(y, unconstrained_x))
			return y

		minimizer.step(closure)
		# after finding a candidate in the unconstrained domain,
		# convert it back to original domain.
		x = transform_to(constraint)(unconstrained_x)
		return x.detach()


	def next_x(self, lower_bound=0, upper_bound=1, num_candidates=5):
		candidates = []
		values = []

		x_init = self.gpmodel.X[-1]
		for i in range(num_candidates):
			x = self.find_a_candidate(x_init, lower_bound, upper_bound)
			y = self.lower_confidence_bound(x)
			candidates.append(x)
			values.append(y)
			x_init = x.new_empty(3).uniform_(lower_bound, upper_bound)

		argmin = torch.min(torch.cat(values), dim=0)[1].item()
		return candidates[argmin]
	




if __name__ == '__main__':
	
	
	data = torch.load('data.pt')

	data = data[:, 0:int(data.size()[1]*2/3)]
	
	candidates = hyperband(54, ((1, 10), (1, 6), (1, 6)), 3, data)

	torch.save(candidates, 'hyperband_3.pt')
	bayesian = bayesian_op()
	candidates = bayesian.run( ((1, 10), (1, 6), (1, 6)), data, 7, 13, 2538)

	torch.save(candidates, 'bayesian_op_3.pt')
	

	



				

	



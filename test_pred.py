import DTSBN
import torch
from torch.nn import MSELoss


def predict_n_forward(DTSBN_net, states, n):
	states_past = states
	states_present = []

	for _ in range(0, n):
		st_top = torch.zeros((DTSBN_net.dims[0], 1))
		st_top[:,0] = DTSBN_net.sample_top_layer_gen(states_past[0], states_past[1])
		states_present.append(st_top.clone())
		for i in range(1, len(states_past)-1):
			st_mid = torch.zeros((DTSBN_net.dims[i], 1))
			st_mid[:,0] = DTSBN_net.sample_mid_layer_gen(states_present[i-1], states_past[i], states_past[i+1], i)
			states_present.append(st_mid.clone())

		st_bot = torch.zeros((DTSBN_net.dims[-1], 1))
		st_bot[:,0] = DTSBN_net.sample_bottom_layer_gen(states_present[-1], states_past[-1])
		states_present.append(st_bot.clone())

		for i in range(0, len(states_present)):

			states_past[i] = torch.cat((states_past[i][:,1:], states_present[i]), 1)

		states_present = []

	return states_past[-1]

if __name__ == '__main__':
	configs = torch.load('hyperband_2.pt')
	config = configs[0]
	data_size = 1404
	print(config)
	steps_back = int(config[3].item())
	levels = int(config[2].item())
	factor = config[1].item()
	dims = [data_size]
	for _ in range(0, levels):
		dims.append(max(1, int(dims[-1]/factor)))

	DDBL_net = DTSBN.DDBL(dims)
	dims.reverse()
	DTSBN_net = DTSBN.DTSBN(dims, levels, steps_back)

	data = torch.load('data.pt')
	train_data = data[:, :int(data.size()[1]*2/3)]

	print(dims)

	train_time = 15
	DTSBN.Adam(DTSBN_net, DDBL_net, train_data, MSELoss(), t_max = 15, Noisy=False)

	test_data = data[:, int(data.size()[1]*2/3): ]

	num_samples = 25
	n_forward = 5
	samples = torch.zeros((test_data.size()[0], test_data.size()[1] - n_forward, num_samples))
	for s in range(0, num_samples):

		preds = torch.zeros((test_data.size()[0], test_data.size()[1] - n_forward))
		data_pred = torch.cat((torch.zeros((data.size()[0], 1)), data), 1)
		states = [data_pred]
		for i in range(0, levels):
			states.append(torch.zeros((dims[-2-i], data.size()[1]+1)))

		for j in range(1, data.size()[1]+1):
			for i in range(1, len(states)):
				s_1 = torch.zeros((states[i-1].size()[0], steps_back))
				s_2 = torch.zeros((states[i-1].size()[0], steps_back))
				s_3 = torch.zeros((states[i].size()[0], steps_back))

				for k in range(0, steps_back):
					s_1[:,k] = states[i-1][:,j -steps_back + k]
					s_2[:,k] = states[i-1][:,j -steps_back + k +1]
					s_3[:,k] = states[i][:,j -steps_back + k]
				states[i][:,j] = DTSBN_net.sample_layer_rec(s_1, s_2, s_3, i-1)


		for d in range(0, test_data.size()[1] - n_forward):
			print(d, s)
			predict_states = []
			for i in range(0, len(states)):
				state = torch.zeros((states[i].size()[0], steps_back))
				for b in range(0, steps_back):
					state[:, b] = states[i][:, d+int(data.size()[1]*2/3)-steps_back+1 + b]
				predict_states.append(state.clone())

			predict_states.reverse()

			preds[:, d] = torch.squeeze(predict_n_forward(DTSBN_net, predict_states, n_forward))
		samples[:,:,s] = preds.clone()



		







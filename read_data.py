import pandas as pd
import torch 

def read_in_data(inpath):
	df = pd.read_csv(inpath)
	d1 = df.groupby(['Name'])['date'].max()
	d1 = d1[d1 == '2018-02-07'].index
	d2 = df.groupby(['Name'])['date'].min()
	d2 = d2[d2 == '2013-02-08'].index

	ticks = [ticker for ticker in d1 if ticker in d2]

	dates_list = []
	for date in df['date'].unique():
		dates_list.append(date)

	dates_list.sort()
	

	data_temp = torch.zeros((4*len(ticks)), len(dates_list))

	t = 0
	for ticker in ticks:
		try:
			data_temp[4*t] = torch.tensor(df.loc[(df['Name'] == ticker)]['close'].values)
			data_temp[4*t+1] = torch.tensor(df.loc[(df['Name'] == ticker)]['high'].values)
			data_temp[4*t+2] = torch.tensor(df.loc[(df['Name'] == ticker)]['low'].values)
			data_temp[4*t+3] = torch.tensor(df.loc[(df['Name'] == ticker)]['volume'].values)
		
			print(t)
			t+=1
		except:
			print(ticker)

	print(t)
	print('PPPPP')

	data_temp = data_temp[0:t*4]
	bad_rows = data_temp.isnan().nonzero()
	restricted_set = set()
	for r in enumerate(bad_rows):
		restricted_set.add(int(r[1][0].item()/4))
		


	data_temp2 = torch.zeros((int(data_temp.size()[0]*3/4)-len(restricted_set)*3, data_temp.size()[1] ))
	
	c_prime = 0
	for c in range(0, t):
		if c not in restricted_set:

			data_temp2[3*c_prime] = data_temp[4*c]
			data_temp2[3*c_prime+1] = torch.sub(data_temp[4*c+1], data_temp[4*c+2])
			data_temp2[3*c_prime+2] = data_temp[4*c+3]
			c_prime += 1

		


	bad_rows = data_temp2.isnan().nonzero()
	
	data = torch.zeros((data_temp2.size()[0], data_temp2.size()[1]-1 ))

	for p in range(1, data_temp2.size()[1]):
		data[:, p-1] = torch.div(torch.sub(data_temp2[:,p], data_temp2[:,p-1]), data_temp2[:,p-1])


	for c in range(0, data.size()[0]):
		data[c] = torch.div(torch.sub(data[c], torch.mean(data[c])), 100*torch.var(data[c]))


	bad_rows = data.isnan().nonzero()
	

	return data







if __name__ == '__main__':
	read_in_data('archive/all_stocks_5yr.csv')
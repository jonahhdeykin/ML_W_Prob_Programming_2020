import pandas as pd
import torch
import pickle
import math


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
    failed = 0
    for ticker in ticks:

        try:
            data_temp[4*t] = torch.tensor(df.loc[(df['Name'] == ticker)]
                                          ['close'].values)
            data_temp[4*t+1] = torch.tensor(df.loc[(df['Name'] == ticker)]
                                            ['high'].values)
            data_temp[4*t+2] = torch.tensor(df.loc[(df['Name'] == ticker)]
                                            ['low'].values)
            data_temp[4*t+3] = torch.tensor(df.loc[(df['Name'] == ticker)]
                                            ['volume'].values)

            print(t)
            t += 1

        except Exception:
            failed += 1

    print(t)

    data_temp = data_temp[0:t*4]
    bad_rows = data_temp.isnan().nonzero()
    restricted_set = set()
    for r in enumerate(bad_rows):
        restricted_set.add(int(r[1][0].item()/4))

    data_temp2 = torch.zeros((int(data_temp.size()[0] * 3/4) -
                              len(restricted_set) * 3, data_temp.size()[1]))

    c_prime = 0
    for c in range(0, t):
        if c not in restricted_set:

            data_temp2[3*c_prime] = data_temp[4*c]
            data_temp2[3*c_prime+1] = torch.sub(data_temp[4*c+1],
                                                data_temp[4*c+2])
            data_temp2[3*c_prime+2] = data_temp[4*c+3]
            c_prime += 1

    data = torch.zeros((data_temp2.size()[0], data_temp2.size()[1]-1))

    for p in range(1, data_temp2.size()[1]):
        data[:, p-1] = torch.div(torch.sub(data_temp2[:, p],
                                 data_temp2[:, p-1]), data_temp2[:, p-1])

    data_recovery = []
    for c in range(0, data.size()[0]):
        data_recovery.append([torch.mean(data[c]), 100*torch.var(data[c])])

        data[c] = torch.div(torch.sub(data[c], torch.mean(data[c])),
                            100*torch.var(data[c]))

    with open('data_recovery.txt', 'wb') as f:
        pickle.dump(data_recovery, f)

    return data


def make_mark_cap(inpath_prices, inpath_cap):

    df = pd.read_csv(inpath_prices)
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

    removal_list = []
    t = 0
    for ticker in ticks:
        try:
            data_temp[4*t] = torch.tensor(df.loc[(df['Name'] == ticker)]
                                          ['close'].values)
            data_temp[4*t+1] = torch.tensor(df.loc[(df['Name'] == ticker)]
                                            ['high'].values)
            data_temp[4*t+2] = torch.tensor(df.loc[(df['Name'] == ticker)]
                                            ['low'].values)
            data_temp[4*t+3] = torch.tensor(df.loc[(df['Name'] == ticker)]
                                            ['volume'].values)

            t += 1
            print(t)
        except Exception:
            removal_list.append(ticker)

    ticks = [ticker for ticker in ticks if ticker not in removal_list]
    data_temp = data_temp[0:t*4]
    bad_rows = data_temp.isnan().nonzero()
    restricted_set = set()

    for r in enumerate(bad_rows):
        restricted_set.add(int(r[1][0].item()/4))

    restricted_list = list(restricted_set)
    restricted_list.sort()
    temp_ticks = []
    temp_ticks += ticks[:restricted_list[0]]
    for r in range(1, len(restricted_list)):
        temp_ticks += ticks[restricted_list[r-1]+1:restricted_list[r]]

    temp_ticks += ticks[restricted_list[-1]+1:]
    ticks = temp_ticks

    with open('price_index_list.txt', 'wb') as f:
        pickle.dump(ticks, f)

    data_temp2 = torch.zeros((int(data_temp.size()[0] * 3/4) -
                              len(restricted_set)*3, data_temp.size()[1]))
    c_prime = 0
    for c in range(0, t):
        if c not in restricted_set:

            data_temp2[3*c_prime] = data_temp[4*c]
            data_temp2[3*c_prime+1] = torch.sub(data_temp[4*c+1],
                                                data_temp[4*c+2])
            data_temp2[3*c_prime+2] = data_temp[4*c+3]
            c_prime += 1

    data = torch.zeros((data_temp2.size()[0], data_temp2.size()[1]-1))
    for p in range(1, data_temp2.size()[1]):
        data[:, p-1] = torch.div(data_temp2[:, p], data_temp2[:, p-1])

    m_cap = pd.read_csv(inpath_cap)

    def make_col(row):
        return str(row['datadate'])[0:4] + '-' + (str(row['datadate'])[4:6] +
                                                  '-' + str(row['datadate'])
                                                  [6:8])

    m_cap['date'] = m_cap.apply(lambda row: make_col(row), 1)
    match_date = []

    removal_list = []
    for ticker in ticks:
        od = m_cap.loc[m_cap['tic'] == ticker]['date'].unique()
        failed = True
        for date in df.loc[df['Name'] == ticker]['date'].unique():
            if date in od and not math.isnan(m_cap.loc[(m_cap['date'] == date)
                                                       & (m_cap['tic'] ==
                                                          ticker)]
                                             ['mkvaltq'].iloc[0]):
                print(ticker)
                match_date.append(date)
                failed = False
                break
        if failed:
            print('failed {}'.format(ticker))
            removal_list.append(ticker)

    cap_data = torch.zeros((int(data_temp2.size()[0]/3)-len(removal_list),
                            data.size()[1]))
    dates = list(df['date'].unique())
    dates.sort()
    with open('dates.txt', 'wb') as f:
        pickle.dump(dates, f)

    rct = 0
    for t in range(0, len(ticks)):

        if ticks[t] not in removal_list:
            start = dates.index(match_date[t-rct])

            cap_data[t-rct][start] = (m_cap.loc[(m_cap['date'] ==
                                                match_date[t-rct]) &
                                                (m_cap['tic']
                                                == ticks[t])]
                                      ['mkvaltq'].iloc[0])

            for tm in range(1, data_temp2.size()[1] - 1 - start):
                cap_data[t - rct][tm + start] = torch.mul(cap_data[t -
                                                                   rct,
                                                                   tm +
                                                                   start - 1],
                                                          data[3 * t]
                                                          [start + tm])

            for tm in range(0, start):

                cap_data[t-rct][start-tm-1] = torch.div(cap_data[t - rct,
                                                                 start-tm],
                                                        data[3*t][start - tm])

            print(t)
        else:
            rct += 1

    ticks = [t for t in ticks if t not in removal_list]
    with open('size_index_list.txt', 'wb') as f:
        pickle.dump(ticks, f)

    return cap_data

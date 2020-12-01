import torch
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm


def predict_n_forward(DTSBN_net, states, n):
    states_past = states
    states_present = []
    movements = torch.zeros((states[-1].size()[0], n))
    for forward in range(0, n):
        st_top = torch.zeros((DTSBN_net.dims[0], 1))
        st_top[:, 0] = DTSBN_net.sample_top_layer_gen(states_past[0],
                                                      states_past[1])
        states_present.append(st_top.clone())
        for st in range(1, len(states_past)-1):
            st_mid = torch.zeros((DTSBN_net.dims[st], 1))
            st_mid[:, 0] = DTSBN_net.sample_mid_layer_gen(states_present
                                                          [st-1],
                                                          states_past[st],
                                                          states_past[st+1],
                                                          st)
            states_present.append(st_mid.clone())
        st_bot = torch.zeros((DTSBN_net.dims[-1], 1))
        st_bot[:, 0] = DTSBN_net.sample_bottom_layer_gen(states_present[-1],
                                                         states_past[-1])
        states_present.append(st_bot.clone())

        for st in range(0, len(states_present)):

            states_past[st] = torch.cat((states_past[st][:, 1:],
                                         states_present[st]), 1)

        states_present = []
        movements[:, forward] = states_past[-1][:, -1]

    return movements


def sample_n_forward(DTSBN_net, data_path, split, n_reps, n_forward,
                     steps_back):
    data = torch.load(data_path)
    test_data = data[:, split:]
    dims = DTSBN_net.dims
    samples = torch.zeros((test_data.size()[0], test_data.size()[1] -
                           n_forward, n_forward, n_reps))
    for rep in range(0, n_reps):

        preds = torch.zeros((test_data.size()[0], test_data.size()[1] -
                             n_forward, n_forward))
        data_pred = torch.cat((torch.zeros((data.size()[0], 1)), data), 1)

        states = [data_pred]
        for layer in range(0, DTSBN_net.n_layers):
            states.append(torch.zeros((dims[-2-layer], data.size()[1]+1)))

        for day in range(1, data.size()[1]+1):
            for st in range(1, len(states)):
                s_1 = torch.zeros((states[st-1].size()[0], steps_back))
                s_2 = torch.zeros((states[st-1].size()[0], steps_back))
                s_3 = torch.zeros((states[st].size()[0], steps_back))

                for sb in range(0, steps_back):
                    s_1[:, sb] = states[st-1][:, day - steps_back + sb]
                    s_2[:, sb] = states[st-1][:, day - steps_back + sb + 1]
                    s_3[:, sb] = states[st][:, day - steps_back + sb]
                states[st][:, day] = DTSBN_net.sample_layer_rec(s_1, s_2, s_3,
                                                                st-1)

        for day in range(0, test_data.size()[1] - n_forward):

            predict_states = []
            for st in range(0, len(states)):
                state = torch.zeros((states[st].size()[0], steps_back))
                for sb in range(0, steps_back):
                    state[:, sb] = states[st][:, day+int(data.size()[1]*2/3)
                                              - steps_back + 1 + sb]
                predict_states.append(state.clone())

            predict_states.reverse()
            preds[:, day] = predict_n_forward(DTSBN_net, predict_states,
                                              n_forward)
        samples[:, :, :, rep] = preds.clone()
        print('Completed rep {} of {}'.format(rep+1, n_reps))
    return samples


def generate_prices(samples, real_prices_path, recovery_info_path, n_forward):
    with open(recovery_info_path, 'rb') as f:
        recovery_info = pickle.load(f)
    real_prices = torch.load(real_prices_path)
    price_predictions = torch.zeros((samples.size()[0], samples.size()[1],
                                     samples.size()[3]))
    movement_predictions = torch.zeros((samples.size()[0], samples.size()[1],
                                        samples.size()[3]))

    for c in range(0, samples.size()[0]):
        samples[c] = torch.add(torch.mul(samples[c], recovery_info[c][1]),
                               recovery_info[c][0])

    for s in range(0, samples.size()[3]):
        for c in range(0, price_predictions.size()[1]):
            index = c - 5 + int((real_prices.size()[1]-1)*2/3)
            current_prices = real_prices[:, index]
            movements = samples[:, c, :, s]
            baseline = torch.ones((current_prices.size()))
            for i in range(0, n_forward):
                baseline = torch.mul(baseline, movements[:, i])
                current_prices = torch.add(torch.mul(movements[:, i],
                                                     current_prices),
                                           current_prices)
            price_predictions[:, c, s] = current_prices
            movement_predictions[:, c, s] = baseline

    return price_predictions


def check_val_weighted_return(actual_path, predictions, n_forward, m_cap_path,
                              index_sample_path, index_m_cap_path):
    actual = torch.load(actual_path)
    actual = actual[:, -predictions.size()[1]-n_forward:]

    m_cap = torch.load(m_cap_path)
    m_cap = m_cap[:, -predictions.size()[1]-n_forward:]

    with open(index_sample_path, 'rb') as f:
        i_s = pickle.load(f)
    with open(index_m_cap_path, 'rb') as f:
        i_m = pickle.load(f)

    index_dict = dict()
    for i in range(0, len(i_m)):
        index_dict[i] = i_s.index(i_m[i])

    predicted_level = torch.zeros((predictions.size()[1],
                                   predictions.size()[2]))

    actual_level = torch.sum(m_cap, 0)

    for j in range(0, predictions.size()[2]):
        for d in range(0, predictions.size()[1]):
            level = 0
            for i in range(0, len(i_m)):
                level += ((predictions[3*index_dict[i], d, j]
                          / actual[3*index_dict[i], d])*m_cap[i, d]).item()
            predicted_level[d, j] = ((level - actual_level[d])
                                     / actual_level[d]).item()

    level_change = torch.zeros(actual_level.size())
    for day in range(n_forward, actual_level.size()[0]):
        level_change[day] = ((actual_level[day]-actual_level[day-n_forward])
                             / actual_level[day-n_forward])

    level_change = level_change[n_forward:]

    return predicted_level, level_change


def graph_rets(actual_d, data_recovery_path, DTSBN_net, n_preds, n_forward):

    actual = actual_d.clone()
    with open(data_recovery_path, 'rb') as f:
        data_recov = pickle.load(f)

    dims = DTSBN_net.dims
    steps_back = DTSBN_net.nt
    states = []
    for st in range(0, len(dims)):
        states.append(torch.zeros((dims[st], steps_back)))

    preds = predict_n_forward(DTSBN_net, states, n_preds)
    preds = preds[:, 100:]

    for comp in range(0, len(data_recov)):

        actual[comp] = torch.add(torch.mul(actual[comp],
                                           data_recov[comp][1]),
                                 data_recov[comp][0])
        preds[comp] = torch.add(torch.mul(preds[comp], data_recov[comp][1]),
                                data_recov[comp][0])

    real_rets = []
    gen_rets = []

    for comp in range(0, len(data_recov)):
        if comp % 3 == 0:

            for start in range(0, actual.size()[1]+1-n_forward):
                r = 1
                for day in range(0, n_forward):
                    r = r*(actual[comp][start+day].item() + 1)
                real_rets.append(r-1)

            for start in range(0, preds.size()[1]+1-n_forward):
                g = 1
                for day in range(0, n_forward):
                    g = g*(preds[comp][start+day].item() + 1)
                gen_rets.append(g-1)

    f1 = sns.distplot(real_rets, hist=False, kde=True,
                      norm_hist=True, label='Actual')
    f1.set_title('{} day mean returns '.format(n_forward))
    f1.set(xlabel='Return', ylabel='PDF')
    sns.distplot(gen_rets, hist=False, kde=True,
                 norm_hist=True, label='Generated')

    plt.legend()
    plt.show()


def graph_vars(actual_d, data_recovery_path, DTSBN_net, n_preds, n_forward):

    actual = actual_d.clone()
    with open(data_recovery_path, 'rb') as f:
        data_recov = pickle.load(f)

    dims = DTSBN_net.dims
    steps_back = DTSBN_net.nt
    states = []
    for st in range(0, len(dims)):
        states.append(torch.zeros((dims[st], steps_back)))

    preds = predict_n_forward(DTSBN_net, states, n_preds)
    preds = preds[:, 100:]

    for comp in range(0, len(data_recov)):

        actual[comp] = torch.add(torch.mul(actual[comp],
                                           data_recov[comp][1]),
                                 data_recov[comp][0])
        preds[comp] = torch.add(torch.mul(preds[comp], data_recov[comp][1]),
                                data_recov[comp][0])

    real_vars = []
    gen_vars = []

    for comp in range(0, len(data_recov)):
        if comp % 3 == 0:
            r_l = []
            g_l = []
            for start in range(0, actual.size()[1]+1-n_forward):
                r = 1
                for day in range(0, n_forward):
                    r = r*(actual[comp][start+day].item() + 1)
                r_l.append(r-1)
            real_vars.append(torch.var(torch.tensor(r_l)).item())
            for start in range(0, preds.size()[1]+1-n_forward):
                g = 1
                for day in range(0, n_forward):
                    g = g*(preds[comp][start+day].item() + 1)
                g_l.append(g-1)
            gen_vars.append(torch.var(torch.tensor(g_l)).item())

    f1 = sns.distplot(real_vars, hist=False, kde=True,
                      norm_hist=True, label='Actual')
    f1.set_title('{} day returns standard deviation'.format(n_forward))
    f1.set(xlabel='Standard Deviation', ylabel='PDF')
    sns.distplot(gen_vars, hist=False, kde=True,
                 norm_hist=True, label='Generated')

    plt.legend()
    plt.show()


def graph_change(predicted, actual, dates, n_forward):
    means = torch.mean(predicted, 1)
    t = [day for day in range(0, means.size()[0])]

    fig, ax = plt.subplots(1)
    ax.plot(t, means, label='Predicted', color='orange')
    ax.plot(t, actual, label='Actual', color='blue')
    plt.xticks([0, int((actual.size()[0]-1)/4),
                int((actual.size()[0]-1)/4)*2,
                int((actual.size()[0]-1)/4)*3, actual.size()[0]-1],
               [dates[-actual.size()[0]],
                dates[-actual.size()[0]+int((actual.size()[0]-1)/4)],
                dates[-actual.size()[0]+int((actual.size()[0]-1)/4)*2],
                dates[-actual.size()[0]+int((actual.size()[0]-1)/4)*3],
                dates[-1]])
    ax.set(xlabel='Date', ylabel='Return')
    ax.set_title('{} Day Returns Predictions'.format(n_forward))
    plt.legend()
    plt.show()


def show_prob(predicted, actual, level):
    sigs = torch.sqrt(torch.var(predicted, 1))
    means = torch.mean(predicted, 1)
    probs = [100*norm.cdf((level - means[day].item())/sigs[day].item())
             for day in range(0, sigs.size()[0])]

    crash_probs = []
    no_crash_probs = []

    for day in range(0, actual.size()[0]):
        if actual[day] < level:
            crash_probs.append(probs[day])
        else:
            no_crash_probs.append(probs[day])

    return (sum(crash_probs)/len(crash_probs),
            sum(no_crash_probs)/len(no_crash_probs))

import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import init_params as ip
import integrate_spmfdm
import custom_lstm
import helper_func as hf

import wandb


def train(cfg, device):
    torch.set_default_dtype(torch.float)  # Set default dtype to float32
    torch.manual_seed(cfg['seed'])  # PyTorch random number generator
    np.random.seed(cfg['seed'])  # Random number generators in other libraries

    'Setting'
    input_size = 2  # number of input for LSTM. [I, Vt]
    output_size = cfg['n_r'] - 1  # number of outputs per NN, i.e., [cs(r=1), ..., cs(r=Nr-1)] for anode or cathode
    l = int(cfg['k'] / cfg['h'])  # time duration to be integrated (k) divided by step size (h) = number of integration

    'Set battery parameters and loss weights'
    p = ip.InitParams(cfg)

    'Import data (concatenated) and the list of data length'
    train_data, train_data_length = hf.load_data_spmfdm(cfg['train_data_type'], cell=cfg['cell_target'])
    val_data, val_data_length = hf.load_data_spmfdm(cfg['val_data_type'], cell=cfg['cell_target'])

    i_train = train_data['Current(A)']  # I > 0 : Discharge,  I < 0 : Charge
    vt_sim_train = train_data['Voltage(V)']
    t_train = train_data['Test_Time(s)']
    t_train = pd.Series(t_train.index.values, name='Test_Time(s)')  # replace by index (to avoid repeated time value)
    i_val = val_data['Current(A)']
    vt_sim_val = val_data['Voltage(V)']
    t_val = val_data['Test_Time(s)']
    t_val = pd.Series(t_val.index.values, name='Test_Time(s)')  # replace by index (to avoid repeated time value)

    css_n_sim_train = train_data['Css_n']
    css_p_sim_train = train_data['Css_p']
    cs_ave_n_sim_train = train_data['Cs_ave_n']
    cs_ave_p_sim_train = train_data['Cs_ave_p']
    css_n_sim_val = val_data['Css_n']
    css_p_sim_val = val_data['Css_p']
    cs_ave_n_sim_val = val_data['Cs_ave_n']
    cs_ave_p_sim_val = val_data['Cs_ave_p']

    'Adding noise and drifting errors'
    i_train_noise = np.random.normal(0, cfg['std_current'] * cfg['noise'], size=i_train.size)
    vt_train_noise = np.random.normal(0, cfg['std_voltage'] * cfg['noise'], size=vt_sim_train.size)
    i_train = i_train + i_train_noise
    vt_sim_train = vt_sim_train + vt_train_noise
    i_val_noise = np.random.normal(0, cfg['std_current'] * cfg['noise'], size=i_val.size)
    vt_val_noise = np.random.normal(0, cfg['std_voltage'] * cfg['noise'], size=vt_sim_val.size)
    i_val = i_val + i_val_noise
    vt_sim_val = vt_sim_val + vt_val_noise

    'Prepare sequential data'
    i_train_seq = torch.from_numpy(hf.create_sequential(i_train, 2 * cfg['k'], train_data_length)).float().to(device)
    vt_sim_train_seq = torch.from_numpy(hf.create_sequential(vt_sim_train, 2 * cfg['k'], train_data_length)).float().to(
        device)
    t_train_seq = torch.from_numpy(hf.create_sequential(t_train, 2 * cfg['k'], train_data_length)).float()
    i_val_seq = torch.from_numpy(hf.create_sequential(i_val, 2 * cfg['k'], val_data_length)).float().to(device)
    vt_sim_val_seq = torch.from_numpy(hf.create_sequential(vt_sim_val, 2 * cfg['k'], val_data_length)).float().to(
        device)
    t_val_seq = torch.from_numpy(hf.create_sequential(t_val, 2 * cfg['k'], val_data_length)).float()

    'Prepare data for true states for validation'
    css_n_sim_train_seq = torch.from_numpy(
        hf.create_sequential(css_n_sim_train, 2 * cfg['k'], train_data_length)).float()
    css_p_sim_train_seq = torch.from_numpy(
        hf.create_sequential(css_p_sim_train, 2 * cfg['k'], train_data_length)).float()
    cs_ave_n_sim_train_seq = torch.from_numpy(
        hf.create_sequential(cs_ave_n_sim_train, 2 * cfg['k'], train_data_length)).float()
    cs_ave_p_sim_train_seq = torch.from_numpy(
        hf.create_sequential(cs_ave_p_sim_train, 2 * cfg['k'], train_data_length)).float()

    'Each dataset has len(data) - 2 * k + 1 sequential datapoints'
    train_data_size = sum(train_data_length) - (2 * cfg['k'] - 1) * len(train_data_length)
    val_data_size = sum(val_data_length) - (2 * cfg['k'] - 1) * len(val_data_length)

    'Use first half (= k time steps) of sequential data for inputs for LSTM layer (to predict initial states)'
    u_train = torch.zeros((train_data_size, cfg['k'], input_size))  # (data_size, sequence, input_size([i, v]))
    u_val = torch.zeros((val_data_size, cfg['k'], input_size))  # (data_size, sequence, input_size([i, v]))
    u_train[:, :, 0] = i_train_seq[:, :cfg['k']]
    u_train[:, :, 1] = vt_sim_train_seq[:, :cfg['k']]
    u_val[:, :, 0] = i_val_seq[:, :cfg['k']]
    u_val[:, :, 1] = vt_sim_val_seq[:, :cfg['k']]

    'Normalize inputs by max-min values of training data'
    ub = torch.ones((u_train.shape[2]))
    lb = torch.zeros((u_train.shape[2]))
    ub[0] = cfg['OneC'] * cfg['max_Crate']
    lb[0] = -cfg['OneC'] * cfg['max_Crate']
    ub[1] = cfg['V_upper']
    lb[1] = cfg['V_lower']
    for j in range(train_data_size):
        u_train[j, :, :] = (u_train[j, :, :] - lb) / (ub - lb)
    for j in range(val_data_size):
        u_val[j, :, :] = (u_val[j, :, :] - lb) / (ub - lb)

    'Use second half (= another k time steps) of sequential data for integration and subsequent loss calculation'
    i_train_seq_sf = i_train_seq[:, cfg['k']:]  # dim [N, k]
    i_train_seq_sf_ave = hf.calc_ave(i_train_seq_sf, cfg['h'])  # average by each h time steps (dim [N, k/h])
    i_train_seq_sampled = hf.sampling(i_train_seq_sf, cfg['h'])  # sampling data at every h
    i_train_seq_sampled = torch.cat((i_train_seq[:, cfg['k'] - 1:cfg['k']], i_train_seq_sampled),
                                    dim=1)  # add the data point to be predicted from NN
    vt_sim_train_seq_sf = vt_sim_train_seq[:, cfg['k']:]
    vt_sim_train_seq_sampled = hf.sampling(vt_sim_train_seq_sf, cfg['h'])  # sampling data at every h
    vt_sim_train_seq_sampled = torch.cat((vt_sim_train_seq[:, cfg['k'] - 1:cfg['k']], vt_sim_train_seq_sampled), dim=1)
    i_val_seq_sf = i_val_seq[:, cfg['k']:]
    i_val_seq_sf_ave = hf.calc_ave(i_val_seq_sf, cfg['h'])
    i_val_seq_sampled = hf.sampling(i_val_seq_sf, cfg['h'])  # sampling data at every h
    i_val_seq_sampled = torch.cat((i_val_seq[:, cfg['k'] - 1:cfg['k']], i_val_seq_sampled),
                                  dim=1)  # add the data point to be predicted from NN
    vt_sim_val_seq_sf = vt_sim_val_seq[:, cfg['k']:]
    vt_sim_val_seq_sampled = hf.sampling(vt_sim_val_seq_sf, cfg['h'])  # sampling data at every h
    vt_sim_val_seq_sampled = torch.cat((vt_sim_val_seq[:, cfg['k'] - 1:cfg['k']], vt_sim_val_seq_sampled), dim=1)

    'Declaring lists for tracing losses'
    losses = np.zeros(cfg['epochs'])
    val_losses = np.zeros(cfg['epochs'])
    nLi_hist = np.zeros(cfg['epochs'])
    R_f_n_hist = np.zeros(cfg['epochs'])
    k_n_hist = np.zeros(cfg['epochs'])
    k_p_hist = np.zeros(cfg['epochs'])
    D_s_n_hist = np.zeros(cfg['epochs'])
    D_s_p_hist = np.zeros(cfg['epochs'])

    last_epoch = -1

    'Defining NN layers'
    fc_layers = np.array(cfg['hidden_lstm'])  # take output of LSTM layer as input for FC layer
    for layer in range(cfg['layer_fc']):
        fc_layers = np.append(fc_layers, cfg['hidden_fc'])
    fc_layers = np.append(fc_layers, output_size)  # output = [xn_1(t+k) ... xn_Q(t+k), xp_1(t+k) ... xp_Q(t+k)]

    'Creating models'
    integrator = integrate_spmfdm.IntegrateSPM(p, cfg['n_r'], cfg['h'], cfg['integ_type'])
    max_values = torch.tensor([p.c_s_n_max, p.c_s_p_max])
    min_values = torch.tensor([0.0, 0.0])

    nn_models = []
    optimizers = []
    for i in range(2):  # each for anode and cathode
        max_value = max_values[i]
        min_value = min_values[i]
        nn_model = custom_lstm.CustomLSTM(cfg, fc_layers, input_size, max_value, min_value)
        nn_models.append(nn_model)
        optimizer = optim.Adam(nn_model.parameters(), lr=cfg['lrate'])
        optimizers.append(optimizer)

    optimizer = optim.Adam(integrator.parameters(), lr=cfg['lrate'])
    optimizers.append(optimizer)
    if cfg['load_nn']:
        nn_models[0].load_state_dict(
            torch.load('training_results/lstm_spmfdm_n_' + cfg['suffix_nn'] + '.pth', map_location=torch.device('cpu')))
        nn_models[1].load_state_dict(
            torch.load('training_results/lstm_spmfdm_p_' + cfg['suffix_nn'] + '.pth', map_location=torch.device('cpu')))
    if cfg['load_pinn']:
        nn_models[0].load_state_dict(
            torch.load('training_results/pilstm_spmfdm_n_' + cfg['suffix_pinn'] + '.pth',
                       map_location=torch.device('cpu')))
        nn_models[1].load_state_dict(
            torch.load('training_results/pilstm_spmfdm_p_' + cfg['suffix_pinn'] + '.pth',
                       map_location=torch.device('cpu')))
        df = pd.read_csv('training_results/pilstm_spmfdm_loss_' + cfg['suffix_pinn'] + '.csv')
        last_epoch = df.nLi[df.nLi != 0].index[-1]
        losses[0:last_epoch + 1] = df.losses[0:last_epoch + 1]
        val_losses[0:last_epoch + 1] = df.val_losses[0:last_epoch + 1]
        nLi_hist[0:last_epoch + 1] = df.nLi[0:last_epoch + 1]
        R_f_n_hist[0:last_epoch + 1] = df.R_f_n[0:last_epoch + 1]
        k_n_hist[0:last_epoch + 1] = df.k_n[0:last_epoch + 1]
        k_p_hist[0:last_epoch + 1] = df.k_p[0:last_epoch + 1]
        D_s_n_hist[0:last_epoch + 1] = df.D_s_n[0:last_epoch + 1]
        D_s_p_hist[0:last_epoch + 1] = df.D_s_p[0:last_epoch + 1]
        integrator.nLi_s_scaled.data = torch.tensor([nLi_hist[last_epoch] / integrator.nLi_s_init]).float()
        integrator.R_f_n_scaled.data = torch.tensor([R_f_n_hist[last_epoch] / integrator.R_f_n_init]).float()
        integrator.k_n_scaled.data = torch.tensor([k_n_hist[last_epoch] / integrator.k_n_init]).float()
        integrator.k_p_scaled.data = torch.tensor([k_p_hist[last_epoch] / integrator.k_p_init]).float()
        integrator.D_s_n_scaled.data = torch.tensor([D_s_n_hist[last_epoch] / integrator.D_s_n_init]).float()
        integrator.D_s_p_scaled.data = torch.tensor([D_s_p_hist[last_epoch] / integrator.D_s_p_init]).float()

    u_train.requires_grad = True
    start_time = time.time()

    mse = nn.MSELoss()

    print('Start training')
    for epoch in range(last_epoch + 1, cfg['epochs']):
        start = 0
        loss_epoch = 0
        for data in train_data_length:
            # selecting data for one current profile
            end = start + data - 2 * cfg['k'] + 1
            u_data = u_train[start:end, :, :]
            i_seq_data = i_train_seq_sampled[start:end, :]
            i_seq_ave_data = i_train_seq_sf_ave[start:end, :]
            vt_sim_seq_data = vt_sim_train_seq_sampled[start:end, :]
            len_data = u_data.shape[0]
            len_batch = int(len_data / cfg['batches'])

            # tensors to store predictions for the current profile over batch iteration
            css_n_data = torch.zeros(len_data)
            css_p_data = torch.zeros(len_data)
            cs_bar_n_data = torch.zeros(len_data)
            cs_bar_p_data = torch.zeros(len_data)
            vt_data = torch.zeros(len_data)

            # variables for storing losses at each batch iteration
            loss_vt_data = 0
            loss_nLi_data = 0
            loss_integ_data = 0

            for batch in range(cfg['batches']):
                # selecting the data for the batch
                batch_start = batch * len_batch
                if batch < cfg['batches'] - 1:
                    batch_end = (batch + 1) * len_batch
                else:
                    batch_end = None
                u_batch = u_data[batch_start:batch_end, :, :]
                i_seq_batch = i_seq_data[batch_start:batch_end, :]
                i_seq_ave_batch = i_seq_ave_data[batch_start:batch_end, :]
                vt_sim_seq_batch = vt_sim_seq_data[batch_start:batch_end, :]
                len_batch = u_batch.shape[0]

                # creating tensors for storing predictions
                x_batch = torch.zeros(len_batch, (cfg['n_r'] - 1) * 2)
                x_seq_batch = torch.zeros(len_batch, (cfg['n_r'] - 1) * 2, l + 1)

                # feed forward pass to make prediction
                for i in range(2):
                    x_batch[:, i * (cfg['n_r'] - 1):(i + 1) * (cfg['n_r'] - 1)] = nn_models[i](u_batch)

                x_seq_batch[:, :, 0] = x_batch

                'Update dynamics based on the latest parameter assumption'
                integrator.update_dynamics()

                # integrate the predicted states over k time steps
                x_seq_batch[:, :, 1:] = integrator(x_batch, i_seq_ave_batch, l)
                'using BC for calculating central and surface concentration and store all the states'
                cs_n_seq_batch, cs_p_seq_batch = integrator.calc_css_and_cs0_seq(x_seq_batch, i_seq_batch)
                css_n_seq_batch = cs_n_seq_batch[:, -1, :]
                css_p_seq_batch = cs_p_seq_batch[:, -1, :]

                cs_bar_n_seq_batch = integrator.spm_model.calc_cs_bar_seq(cs_n_seq_batch)  # average concentration
                cs_bar_p_seq_batch = integrator.spm_model.calc_cs_bar_seq(cs_p_seq_batch)  # average concentration

                'Loss from voltage error'
                vt_seq_batch = integrator.spm_model.calc_voltage(css_n_seq_batch, css_p_seq_batch, p.c_e, i_seq_batch,
                                                                 integrator.p.k_n, integrator.p.k_p, integrator.p.R_f_n)
                loss_vt = torch.sqrt(mse(vt_seq_batch, vt_sim_seq_batch))

                'Loss from lithium conservation violation'
                nLi_n_seq = cs_bar_n_seq_batch * p.epsilon_s_n * p.L_n * p.A_n
                nLi_p_seq = cs_bar_p_seq_batch * p.epsilon_s_p * p.L_p * p.A_p
                nLi_seq = nLi_n_seq + nLi_p_seq
                loss_nLi = torch.sqrt(mse(nLi_seq, integrator.p.nLi_s * torch.ones(nLi_seq.shape)))

                'Calculate loss if NN(I(t+k)) deviate from NN(I(t)) integrated over k time steps'
                loss_integ = 0
                for i in range(1, l):
                    cs_n_k_normalized = cs_n_seq_batch[0:-l, :, i] / p.c_s_n_max
                    cs_p_k_normalized = cs_p_seq_batch[0:-l, :, i] / p.c_s_p_max
                    cs_n_0_integrated_normalized = cs_n_seq_batch[i:-(l - i), :, 0] / p.c_s_n_max
                    cs_p_0_integrated_normalized = cs_p_seq_batch[i:-(l - i), :, 0] / p.c_s_p_max
                    loss_integ_n = torch.sqrt(mse(cs_n_k_normalized, cs_n_0_integrated_normalized) / l)
                    loss_integ_p = torch.sqrt(mse(cs_p_k_normalized, cs_p_0_integrated_normalized) / l)
                    loss_integ += loss_integ_n + loss_integ_p

                'Calculate total loss and update'
                loss = loss_nLi + loss_vt + loss_integ
                loss.backward()  # Does backpropagation and calculates gradients
                for optimizer in optimizers:
                    optimizer.step()  # Updates the weights accordingly
                    optimizer.zero_grad()

                for params in integrator.parameters():
                    params.data.clamp_(cfg['p_search_lower'],
                                       cfg['p_search_upper'])  # constraint params between 67% and 150% of original

                # storing result for printing/visualization purpose
                loss_vt_data += loss_vt.item()
                loss_nLi_data += loss_nLi.item()
                loss_integ_data += loss_integ
                css_n_data[batch_start:batch_end] = css_n_seq_batch[:, 0]
                css_p_data[batch_start:batch_end] = css_p_seq_batch[:, 0]
                cs_bar_n_data[batch_start:batch_end] = cs_bar_n_seq_batch[:, 0]
                cs_bar_p_data[batch_start:batch_end] = cs_bar_p_seq_batch[:, 0]
                vt_data[batch_start:batch_end] = vt_seq_batch[:, 0]

            loss_vt_data = loss_vt_data / cfg['batches']
            loss_nLi_data = loss_nLi_data / cfg['batches']
            loss_integ_data = loss_integ_data / cfg['batches']
            loss_data = loss_vt_data + loss_nLi_data + loss_integ_data

            if epoch % 25 == 0:
                print(f'Finished epoch {epoch}, training loss {loss_data} (from vt: {loss_vt_data}, '
                      f'from nLi: {loss_nLi_data}, from integ: {loss_integ_data})\n'
                      f'param error: nLi {np.round(100 * (integrator.p.nLi_s.detach().cpu().numpy() / p.nLi_s_true - 1))[0]}%, '
                      f'R_f_n {np.round(100 * (integrator.p.R_f_n.detach().cpu().numpy() / p.R_f_n_true - 1))[0]}%, '
                      f'k_n {np.round(100 * (integrator.p.k_n.detach().cpu().numpy() / p.k_n_true - 1))[0]}%, '
                      f'k_p {np.round(100 * (integrator.p.k_p.detach().cpu().numpy() / p.k_p_true - 1))[0]}%, '
                      f'D_s_n {np.round(100 * (integrator.p.D_s_n.detach().cpu().numpy() / p.D_s_n_true - 1))[0]}%, '
                      f'D_s_p {np.round(100 * (integrator.p.D_s_p.detach().cpu().numpy() / p.D_s_p_true - 1))[0]}% \n')

            if epoch % 1000 == 1 or epoch == cfg['epochs'] - 1:
                css_n_sim = css_n_sim_train_seq[start:end, cfg['k'] - 1]
                cs_ave_n_sim = cs_ave_n_sim_train_seq[start:end, cfg['k'] - 1]
                css_p_sim = css_p_sim_train_seq[start:end, cfg['k'] - 1]
                cs_ave_p_sim = cs_ave_p_sim_train_seq[start:end, cfg['k'] - 1]
                vt_sim = vt_sim_train_seq[start:end, cfg['k'] - 1].detach().cpu().numpy()
                t_data = t_train_seq[start:end, cfg['k'] - 1]
                nLi_est = (integrator.p.nLi_s * torch.ones(t_data.shape)).detach().cpu().numpy()

                _, ax = plt.subplots(3, 4, figsize=((end - start) / 150, 12))
                hf.set_fig2(ax, 0, 0, t_data, css_n_data.detach().cpu().numpy(), css_n_sim, 'Css_n')
                hf.set_fig2(ax, 1, 0, t_data, cs_bar_n_data.detach().cpu().numpy(), cs_ave_n_sim, 'Cs_ave_n')
                hf.set_fig2(ax, 0, 1, t_data, css_p_data.detach().cpu().numpy(), css_p_sim, 'Css_p')
                hf.set_fig2(ax, 1, 1, t_data, cs_bar_p_data.detach().cpu().numpy(), cs_ave_p_sim, 'Cs_ave_p')
                hf.set_fig2(ax, 0, 2, t_data, vt_data.detach().cpu().numpy(), vt_sim, 'Voltage')
                hf.set_fig(ax, 1, 2, losses[:epoch], 'epoch', 'loss', bottom=0.02, top=0.2)
                ax[1, 2].set_yscale("log")
                hf.set_fig(ax, 0, 3, nLi_hist[:epoch], 'epoch', 'nLi', p.nLi_s_true * np.ones(losses[:epoch].size))
                hf.set_fig(ax, 1, 3, R_f_n_hist[:epoch], 'epoch', 'R_f_n', p.R_f_n_true * np.ones(losses[:epoch].size))
                hf.set_fig(ax, 2, 0, k_n_hist[:epoch], 'epoch', 'k_n', p.k_n_true * np.ones(losses[:epoch].size))
                hf.set_fig(ax, 2, 1, k_p_hist[:epoch], 'epoch', 'k_p', p.k_p_true * np.ones(losses[:epoch].size))
                hf.set_fig(ax, 2, 2, D_s_n_hist[:epoch], 'epoch', 'D_s_n', p.D_s_n_true * np.ones(losses[:epoch].size))
                hf.set_fig(ax, 2, 3, D_s_p_hist[:epoch], 'epoch', 'D_s_p', p.D_s_p_true * np.ones(losses[:epoch].size))
                plt.suptitle(f"Estimated Initial conditions for training data, lr= {cfg['lrate']} at epoch {epoch} "
                             f"(N_r = {cfg['n_r']}, k = {cfg['k']}, LSTM size = {cfg['hidden_lstm']}, FC size = {cfg['hidden_fc']}, "
                             f"noise = {cfg['noise']}, step size = {cfg['h']})")
                plt.show()
                if cfg['save']:
                    torch.save(nn_models[0].state_dict(),
                               'training_results/pilstm_spmfdm_n_' + cfg['suffix_save'] + '.pth')
                    torch.save(nn_models[1].state_dict(),
                               'training_results/pilstm_spmfdm_p_' + cfg['suffix_save'] + '.pth')
                    df = pd.DataFrame({"losses": losses, "nLi": nLi_hist, "R_f_n": R_f_n_hist, "k_n": k_n_hist,
                                       "k_p": k_p_hist, "D_s_n": D_s_n_hist, "D_s_p": D_s_p_hist,
                                       "val_losses": val_losses})
                    df.to_csv('training_results/pilstm_spmfdm_loss_' + cfg['suffix_save'] + '.csv', index=False)

            start = end
            loss_epoch += loss_data

        losses[epoch] = loss_epoch
        nLi_hist[epoch] = integrator.p.nLi_s
        R_f_n_hist[epoch] = integrator.p.R_f_n
        k_n_hist[epoch] = integrator.p.k_n
        k_p_hist[epoch] = integrator.p.k_p
        D_s_n_hist[epoch] = integrator.p.D_s_n
        D_s_p_hist[epoch] = integrator.p.D_s_p

        if epoch % 1000 == 1 or epoch == cfg['epochs'] - 1:
            max_batch_size = 2000
            with torch.no_grad():
                x_val = torch.zeros(u_val.shape[0], (cfg['n_r'] - 1) * 2)
                x_seq_val = torch.zeros(u_val.shape[0], (cfg['n_r'] - 1) * 2, l + 1)

                for i in range(2):
                    state_start = i * (cfg['n_r'] - 1)
                    state_end = (i + 1) * (cfg['n_r'] - 1)
                    data_start = 0
                    for data in val_data_length:
                        data_end = data_start + data - 2 * cfg['k'] + 1
                        batch_start = data_start
                        batch_end = data_start
                        while batch_end < data_end:
                            batch_end += max_batch_size
                            batch_end = min(batch_end, data_end)
                            x_val[batch_start:batch_end, state_start:state_end] = nn_models[i](
                                u_val[batch_start:batch_end])
                            batch_start = batch_end
                        data_start = data_end

                x_seq_val[:, :, 0] = x_val
                x_seq_val[:, :, 1:] = integrator(x_val, i_val_seq_sf_ave, l)

                cs_n_seq_val, cs_p_seq_val = integrator.calc_css_and_cs0_seq(x_seq_val, i_val_seq_sampled)
                css_n_seq_val = cs_n_seq_val[:, -1]
                css_p_seq_val = cs_p_seq_val[:, -1]

                cs_bar_n_seq_val = integrator.spm_model.calc_cs_bar_seq(cs_n_seq_val)
                cs_bar_p_seq_val = integrator.spm_model.calc_cs_bar_seq(cs_p_seq_val)

                vt_seq_val = integrator.spm_model.calc_voltage(css_n_seq_val, css_p_seq_val, p.c_e,
                                                               i_val_seq_sampled,
                                                               integrator.p.k_n, integrator.p.k_p,
                                                               integrator.p.R_f_n)
                loss_vt_val = torch.sqrt(mse(vt_seq_val, vt_sim_val_seq_sampled))

                'Calculate loss if NN(I(t+k)) deviate from NN(I(t)) integrated over k time steps'
                loss_integ_val = 0
                for i in range(1, l):
                    cs_n_k_normalized = cs_n_seq_val[0:-l, :, i] / p.c_s_n_max
                    cs_p_k_normalized = cs_p_seq_val[0:-l, :, i] / p.c_s_p_max
                    cs_n_0_integrated_normalized = cs_n_seq_val[i:-(l - i), :, 0] / p.c_s_n_max
                    cs_p_0_integrated_normalized = cs_p_seq_val[i:-(l - i), :, 0] / p.c_s_p_max
                    loss_integ_n = torch.sqrt(mse(cs_n_k_normalized, cs_n_0_integrated_normalized) / l)
                    loss_integ_p = torch.sqrt(mse(cs_p_k_normalized, cs_p_0_integrated_normalized) / l)
                    loss_integ_val += loss_integ_n + loss_integ_p

                nLi_n_seq = cs_bar_n_seq_val * p.epsilon_s_n * p.L_n * p.A_n
                nLi_p_seq = cs_bar_p_seq_val * p.epsilon_s_p * p.L_p * p.A_p
                nLi_seq = nLi_n_seq + nLi_p_seq
                loss_nLi_val = torch.sqrt(mse(nLi_seq, integrator.p.nLi_s * torch.ones(nLi_seq.shape)))

                loss_val = loss_vt_val + loss_nLi_val + loss_integ_val

                val_losses[epoch] = loss_val.item()

            'Plots for validation data'
            _, ax = plt.subplots(3, 2, figsize=(50, 10), layout="constrained")
            hf.set_fig2(ax, 0, 0, t_val_seq[:, cfg['k'] - 1], css_n_seq_val[:, 0].detach().cpu().numpy(),
                        css_n_sim_val[cfg['k']:-cfg['k'] + 1], 'Css_n')
            hf.set_fig2(ax, 0, 1, t_val_seq[:, cfg['k'] - 1], css_p_seq_val[:, 0].detach().cpu().numpy(),
                        css_p_sim_val[cfg['k']:-cfg['k'] + 1], 'Css_p')
            hf.set_fig2(ax, 1, 0, t_val_seq[:, cfg['k'] - 1], cs_bar_n_seq_val[:, 0].detach().cpu().numpy(),
                        cs_ave_n_sim_val[cfg['k']:-cfg['k'] + 1], 'Cs_ave_n')
            hf.set_fig2(ax, 1, 1, t_val_seq[:, cfg['k'] - 1], cs_bar_p_seq_val[:, 0].detach().cpu().numpy(),
                        cs_ave_p_sim_val[cfg['k']:-cfg['k'] + 1], 'Cs_ave_p')
            hf.set_fig2(ax, 2, 0, t_val_seq[:, cfg['k'] - 1], vt_seq_val[:, 0].detach().cpu().numpy(),
                        vt_sim_val[cfg['k']:-cfg['k'] + 1], 'Vt')
            hf.set_fig(ax, 2, 1, losses[:epoch], 'epoch', 'loss', val_losses[:epoch], 'training', 'validation',
                       bottom=0.01, top=1)
            ax[2, 1].set_yscale("log")
            plt.suptitle(f"Estimated Initial conditions for validation data, lr= {cfg['lrate']}")
            plt.show()

    elapsed = time.time() - start_time
    print('Training time: %.2f' % elapsed)

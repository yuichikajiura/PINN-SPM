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



def train(cfg, device):
    torch.set_default_dtype(torch.float)  # Set default dtype to float32
    torch.manual_seed(cfg['seed'])  # PyTorch random number generator
    np.random.seed(cfg['seed'])  # Random number generators in other libraries

    'Setting'
    input_size = 2  # [I, Vt]
    output_size = cfg['n_r'] - 1  # number of outputs per NN, i.e., [cs(r=1), ..., cs(r=Nr-1)] for anode or cathode
    k = cfg['k']

    'Set battery parameters and loss weights (TBD: need to modify parameter for cell 2)'
    p = ip.InitParams(cfg)

    'Import data (concatenated) and the list of data length'
    train_data, train_data_length = hf.load_data_spmfdm(cfg['train_data_type'], states=True, cell=cfg['cell_target'])
    val_data, val_data_length = hf.load_data_spmfdm(cfg['val_data_type'], states=True, cell=cfg['cell_target'])

    cn_train = train_data.iloc[:, 7:28]
    cp_train = train_data.iloc[:, 28:]
    cnx_train = cn_train.iloc[:, 1:-1]  # selecting except cp,0 and cp,s
    cpx_train = cp_train.iloc[:, 1:-1]  # selecting except cp,0 and cp,s
    cx_train = pd.concat([cnx_train, cpx_train], axis=1)
    cn_val = val_data.iloc[:, 7:28]
    cp_val = val_data.iloc[:, 28:]
    cnx_val = cn_val.iloc[:, 1:-1]  # selecting except cp,0 and cp,s
    cpx_val = cp_val.iloc[:, 1:-1]  # selecting except cp,0 and cp,s
    cx_val = pd.concat([cnx_val, cpx_val], axis=1)

    i_train = train_data.iloc[:, 1]  # I > 0 : Discharge,  I < 0 : Charge
    vt_sim_train = train_data.iloc[:, 2]
    t_train = train_data.iloc[:, 0]
    t_train = pd.Series(t_train.index.values, name='Test_Time(s)')  # replace by index (to avoid repeated time value)
    i_val = val_data.iloc[:, 1]  # I > 0 : Discharge,  I < 0 : Charge
    vt_sim_val = val_data.iloc[:, 2]
    t_val = val_data.iloc[:, 0]
    t_val = pd.Series(t_val.index.values, name='Test_Time(s)')  # replace by index (to avoid repeated time value)

    css_n_sim_train = train_data.iloc[:, 3]
    css_p_sim_train = train_data.iloc[:, 5]
    cs_ave_n_sim_train = train_data.iloc[:, 4]
    cs_ave_p_sim_train = train_data.iloc[:, 6]
    css_n_sim_val = val_data.iloc[:, 3]
    css_p_sim_val = val_data.iloc[:, 5]
    cs_ave_n_sim_val = val_data.iloc[:, 4]
    cs_ave_p_sim_val = val_data.iloc[:, 6]

    'Prepare sequential data'
    i_train_seq = torch.from_numpy(hf.create_sequential(i_train, 2 * k, train_data_length)).float().to(device)
    vt_sim_train_seq = torch.from_numpy(hf.create_sequential(vt_sim_train, 2 * k, train_data_length)).float().to(device)
    t_train_seq = torch.from_numpy(hf.create_sequential(t_train, 2 * k, train_data_length)).float()
    i_val_seq = torch.from_numpy(hf.create_sequential(i_val, 2 * k, val_data_length)).float().to(device)
    vt_sim_val_seq = torch.from_numpy(hf.create_sequential(vt_sim_val, 2 * k, val_data_length)).float().to(device)
    t_val_seq = torch.from_numpy(hf.create_sequential(t_val, 2 * k, val_data_length)).float()

    'Prepare data for true states for validation'
    css_n_sim_train_seq = torch.from_numpy(hf.create_sequential(css_n_sim_train, 2 * k, train_data_length)).float()
    css_p_sim_train_seq = torch.from_numpy(hf.create_sequential(css_p_sim_train, 2 * k, train_data_length)).float()
    cs_ave_n_sim_train_seq = torch.from_numpy(hf.create_sequential(cs_ave_n_sim_train, 2 * k, train_data_length)).float()
    cs_ave_p_sim_train_seq = torch.from_numpy(hf.create_sequential(cs_ave_p_sim_train, 2 * k, train_data_length)).float()
    css_n_sim_val_seq = torch.from_numpy(hf.create_sequential(css_n_sim_val, 2 * k, val_data_length)).float()
    css_p_sim_val_seq = torch.from_numpy(hf.create_sequential(css_p_sim_val, 2 * k, val_data_length)).float()
    cs_ave_n_sim_val_seq = torch.from_numpy(hf.create_sequential(cs_ave_n_sim_val, 2 * k, val_data_length)).float()
    cs_ave_p_sim_val_seq = torch.from_numpy(hf.create_sequential(cs_ave_p_sim_val, 2 * k, val_data_length)).float()

    'Each dataset has len(data) - 2 * k + 1 sequential datapoints'
    train_data_size = sum(train_data_length) - (2 * k - 1) * len(train_data_length)
    val_data_size = sum(val_data_length) - (2 * k - 1) * len(val_data_length)

    'Use first half (= k time steps) of sequential data for inputs for LSTM layer (to predict initial states)'
    u_train = torch.zeros((train_data_size, k, input_size))  # (data_size, sequence, input_size([i, v]))
    u_val = torch.zeros((val_data_size, k, input_size))  # (data_size, sequence, input_size([i, v]))
    u_train[:, :, 0] = i_train_seq[:, :k]
    u_train[:, :, 1] = vt_sim_train_seq[:, :k]
    u_val[:, :, 0] = i_val_seq[:, :k]
    u_val[:, :, 1] = vt_sim_val_seq[:, :k]

    'Normalize inputs by max-min values of training data'
    ub = torch.ones((u_train.shape[2]))
    lb = torch.zeros((u_train.shape[2]))
    ub[0] = cfg['OneC'] * cfg['max_Crate']
    lb[0] = - cfg['OneC'] * cfg['max_Crate']
    ub[1] = cfg['V_upper']
    lb[1] = cfg['V_lower']
    for j in range(train_data_size):
        u_train[j, :, :] = (u_train[j, :, :] - lb) / (ub - lb)
    for j in range(val_data_size):
        u_val[j, :, :] = (u_val[j, :, :] - lb) / (ub - lb)

    'Declaring lists for tracing losses'
    losses = np.zeros(cfg['epochs'])
    last_epoch = -1

    'Enable when loading saved models'
    if cfg['load']:
        df = pd.read_csv('training_results/pilstm_spmfdm_' + str(k) + 'k_loss_'+cfg['suffix']+'.csv')
        last_epoch = df.losses[df.losses != 0].index[-1]
        losses[0:last_epoch + 1] = df.losses[0:last_epoch + 1]

    'Defining NN layers'
    fc_layers = np.array(cfg['hidden_lstm'])  # take output of LSTM layer as input for FC layer
    for layer in range(cfg['layer_fc']):
        fc_layers = np.append(fc_layers, cfg['hidden_fc'])
    fc_layers = np.append(fc_layers, output_size)  # output = [xn_1(t+k) ... xn_Q(t+k), xp_1(t+k) ... xp_Q(t+k)]

    'Creating models'
    integrator = integrate_spmfdm.IntegrateSPM(p, cfg)
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

    if cfg['load']:
        nn_models[0].load_state_dict(torch.load('training_results/pilstm_spmfdm_n_' + cfg['suffix'] + '.pth', map_location=torch.device('cpu')))
        nn_models[1].load_state_dict(torch.load('training_results/pilstm_spmfdm_p_' + cfg['suffix'] + '.pth', map_location=torch.device('cpu')))

    u_train.requires_grad = True
    start_time = time.time()

    mse = nn.MSELoss()

    print('Start training')
    for epoch in range(last_epoch + 1, cfg['epochs']):
        start = 0
        start_seq = 0
        loss_epoch = 0
        for data in train_data_length:
            # selecting data for one current profile
            end_seq = start_seq + data - 2 * k + 1
            end = start + data
            u_data = u_train[start_seq:end_seq, :, :]
            i_data = i_train_seq[start_seq:end_seq, k-1]
            vt_sim_data = vt_sim_train_seq[start_seq:end_seq, k-1]
            len_data = u_data.shape[0]
            len_batch = int(len_data / cfg['batches'])
            x_data_true = cx_train.iloc[start:end, :]
            x_data_true = x_data_true.iloc[k-1:-k,:]  # select data in the time steps to be predicted from rnn
            x_data_true = torch.tensor(x_data_true.values).float()

            # tensors to store predictions for the current profile over batch iteration
            css_n_data = torch.zeros(len_data)
            css_p_data = torch.zeros(len_data)
            cs_bar_n_data = torch.zeros(len_data)
            cs_bar_p_data = torch.zeros(len_data)
            vt_data = torch.zeros(len_data)
            nLi_data = torch.zeros(len_data)

            #variables for storing losses at each batch iteration
            loss_data = 0

            for batch in range(cfg['batches']):
                # selecting the data for the batch
                batch_start = batch*len_batch
                if batch < cfg['batches'] - 1:
                    batch_end = (batch+1)*len_batch
                else:
                    batch_end = None
                u_batch = u_data[batch_start:batch_end, :, :]
                i_batch = i_data[batch_start:batch_end]
                len_batch = u_batch.shape[0]

                # creating tensors for storing predictions
                x_batch = torch.zeros(len_batch, (cfg['n_r'] - 1) * 2)

                # feed forward pass to make prediction
                for i in range(2):
                    x_batch[:, i * (cfg['n_r'] - 1):(i + 1) * (cfg['n_r'] - 1)] = nn_models[i](u_batch)

                # for testing purpose
                loss_x = torch.sqrt(mse(x_batch, x_data_true[batch_start:batch_end, :]))

                'Calculating central and surface concentration from NN prediction'
                cs_n_batch, cs_p_batch = integrator.calc_css_and_cs0(x_batch, i_batch)
                css_n_batch = cs_n_batch[:, -1]
                css_p_batch = cs_p_batch[:, -1]

                vt_batch = integrator.spm_model.calc_voltage(css_n_batch, css_p_batch, p.c_e, i_batch, p.k_n, p.k_p, p.R_f_n)

                'Correcting state prediction such that predicted nLi aligns with the (estimated) true nLi'
                cs_bar_n_batch = integrator.spm_model.calc_cs_bar(cs_n_batch)  # average concentration
                cs_bar_p_batch = integrator.spm_model.calc_cs_bar(cs_p_batch)  # average concentration
                nLi_n = cs_bar_n_batch * p.epsilon_s_n * p.L_n * p.A_n
                nLi_p = cs_bar_p_batch * p.epsilon_s_p * p.L_p * p.A_p
                nLi = nLi_n + nLi_p
                nLi = nLi.unsqueeze(1)

                'Calculate total loss and update'
                loss_batch = loss_x
                loss_batch.backward()  # Does backpropagation and calculates gradients
                for optimizer in optimizers:
                    optimizer.step()  # Updates the weights accordingly
                    optimizer.zero_grad()


                # storing result for printing/visualization purpose
                css_n_data[batch_start:batch_end] = css_n_batch
                css_p_data[batch_start:batch_end] = css_p_batch
                cs_bar_n_data[batch_start:batch_end] = cs_bar_n_batch
                cs_bar_p_data[batch_start:batch_end] = cs_bar_p_batch
                vt_data[batch_start:batch_end] = vt_batch
                nLi_data[batch_start:batch_end] = nLi.squeeze(1)
                loss_data += loss_batch.data

            if epoch % 100 == 0:
                print(f'Finished epoch {epoch}, training loss {loss_data}')

            if epoch % 5000 == 1 or epoch == cfg['epochs'] - 1:
                css_n_sim = css_n_sim_train_seq[start_seq:end_seq, k-1]
                cs_ave_n_sim = cs_ave_n_sim_train_seq[start_seq:end_seq, k-1]
                css_p_sim = css_p_sim_train_seq[start_seq:end_seq, k-1]
                cs_ave_p_sim = cs_ave_p_sim_train_seq[start_seq:end_seq, k-1]
                vt_sim = vt_sim_train_seq[start_seq:end_seq, k-1].detach().cpu().numpy()
                t_data = t_train_seq[start_seq:end_seq, k-1]

                _, ax = plt.subplots(2, 4, figsize=((end_seq-start_seq)/150, 12))
                hf.set_fig2(ax, 0, 0, t_data, css_n_data.detach().cpu().numpy(), css_n_sim, 'Css_n')
                hf.set_fig2(ax, 1, 0, t_data, cs_bar_n_data.detach().cpu().numpy(), cs_ave_n_sim, 'Cs_ave_n')
                hf.set_fig2(ax, 0, 1, t_data, css_p_data.detach().cpu().numpy(), css_p_sim, 'Css_p')
                hf.set_fig2(ax, 1, 1, t_data, cs_bar_p_data.detach().cpu().numpy(), cs_ave_p_sim, 'Cs_ave_p')
                hf.set_fig2(ax, 0, 2, t_data, vt_data.detach().cpu().numpy(), vt_sim, 'Voltage')
                hf.set_fig(ax, 0, 3, losses[:epoch], 'epoch', 'loss')
                ax[1, 3].set_yscale("log")
                plt.suptitle(f"Estimated Initial conditions for training data, lr= {cfg['lrate']} at epoch {epoch} "
                             f"(N_r = {cfg['n_r']}, k = {k}, LSTM size = {cfg['hidden_lstm']}, FC size = {cfg['hidden_fc']}, "
                             f"noise = {cfg['noise']}, step size = {cfg['h']})")
                plt.show()
                if cfg['save']:
                    torch.save(nn_models[0].state_dict(), 'training_results/lstm_spmfdm_n_' + cfg['suffix'] + '.pth')
                    torch.save(nn_models[1].state_dict(), 'training_results/lstm_spmfdm_p_' + cfg['suffix'] + '.pth')
                    df = pd.DataFrame({"losses": losses})
                    df.to_csv('training_results/lstm_spmfdm_' + cfg['suffix'] + '.csv', index=False)

            start = end
            start_seq = end_seq
            loss_epoch += loss_data

        losses[epoch] = loss_epoch

    elapsed = time.time() - start_time
    print('Training time: %.2f' % elapsed)

    max_batch_size = 2000
    with torch.no_grad():
        x_train = torch.zeros(u_train.shape[0], (cfg['n_r'] - 1) * 2)
        x_val = torch.zeros(u_val.shape[0], (cfg['n_r'] - 1) * 2)

        for i in range(2):
            state_start = i * (cfg['n_r'] - 1)
            state_end = (i + 1) * (cfg['n_r'] - 1)
            start_seq = 0
            for data in train_data_length:
                end_seq = start_seq + data - 2 * k + 1
                batch_start = start_seq
                batch_end = start_seq
                while batch_end < end_seq:
                    batch_end += max_batch_size
                    batch_end = min(batch_end, end_seq)
                    x_train[batch_start:batch_end, state_start:state_end] = nn_models[i](u_train[batch_start:batch_end])
                    batch_start = batch_end
                start_seq = end_seq
            start_seq = 0
            for data in val_data_length:
                end_seq = start_seq + data - 2 * k + 1
                batch_start = start_seq
                batch_end = start_seq
                while batch_end < end_seq:
                    batch_end += max_batch_size
                    batch_end = min(batch_end, end_seq)
                    x_val[batch_start:batch_end, state_start:state_end] = nn_models[i](u_val[batch_start:batch_end])
                    batch_start = batch_end
                start_seq = end_seq
        i_val = i_val_seq[:, k-1]
        cs_n_val, cs_p_val = integrator.calc_css_and_cs0(x_val, i_val)
        css_n_val = cs_n_val[:, -1]
        css_p_val = cs_p_val[:, -1]
        cs_bar_n_val = integrator.spm_model.calc_cs_bar(cs_n_val)  # average concentration
        cs_bar_p_val = integrator.spm_model.calc_cs_bar(cs_p_val)  # average concentration
        vt_val = integrator.spm_model.calc_voltage(css_n_val, css_p_val, p.c_e, i_val, p.k_n, p.k_p, p.R_f_n)
        t_val = t_val_seq[:, k-1]

        _, ax = plt.subplots(2, 3, figsize=(18, 12))
        hf.set_fig2(ax, 0, 0, t_val, css_n_sim_val_seq[:, k - 1], css_n_val.detach().cpu().numpy(),  'Css_n')
        hf.set_fig2(ax, 1, 0, t_val, cs_ave_n_sim_val_seq[:, k - 1], cs_bar_n_val.detach().cpu().numpy(),  'Cs_ave_n')
        hf.set_fig2(ax, 0, 1, t_val, css_p_sim_val_seq[:, k - 1], css_p_val.detach().cpu().numpy(), 'Css_p')
        hf.set_fig2(ax, 1, 1, t_val, cs_ave_p_sim_val_seq[:, k - 1], cs_bar_p_val.detach().cpu().numpy(), 'Cs_ave_p')
        hf.set_fig2(ax, 0, 2, t_val, vt_sim_val_seq[:, k - 1], vt_val.detach().cpu().numpy(), 'Voltage')
        hf.set_fig(ax, 1, 2, losses, 'epoch', 'loss')
        plt.suptitle(f"Estimated Initial conditions for validation data"
                     f"(N_r = {cfg['n_r']}, k = {k}, LSTM size = {cfg['hidden_lstm']}, FC size = {cfg['hidden_fc']})")
        plt.show()






import pandas as pd
import numpy as np
import torch


def ocv7(z):
    return (179.3018624 * z ** 7
            - 662.9843922 * z ** 6
            + 988.4086711 * z ** 5
            - 763.3324119 * z ** 4
            + 327.4461809 * z ** 3
            - 77.31237098 * z ** 2
            + 9.620312047 * z
            + 3.039475779)


def set_fig(ax, row, col, y1, xlabel, ylabel, y2=None, y1label=None, y2label=None, bottom=None, top=None):
    ax[row, col].plot(y1, label=y1label)
    if y2 is not None:
        ax[row, col].plot(y2, label=y2label)
    ax[row, col].set_xlabel(xlabel)
    ax[row, col].set_ylabel(ylabel)
    ax[row, col].set_ylim(bottom=bottom, top=top)
    ax[row, col].legend()


def set_fig2(ax, row, col, x, y_est, y_true, ylabel, legend1='Estimated', legend2='True'):
    ax[row, col].plot(x, y_est, label=legend1, linewidth=0.5)
    ax[row, col].plot(x, y_true, label=legend2, linewidth=0.5)
    ax[row, col].set_xlabel('time[s]')
    ax[row, col].set_ylabel(ylabel)
    ax[row, col].legend()


def set_scatter(ax, row, col, x, y, xlabel, ylabel, with_line=True):
    ax[row, col].scatter(x, y, s=0.5, label='estimated')
    ax[row, col].set_xlabel(xlabel)
    ax[row, col].set_ylabel(ylabel)
    if with_line:
        ax[row, col].plot([np.mean((x.min(), y.min())), np.mean((x.max(), y.max()))],
                          [np.mean((x.min(), y.min())), np.mean((x.max(), y.max()))], color='red')


def load_data(data_type):
    df = pd.DataFrame()
    length = []
    for j in data_type:
        if j == 0:
            data = pd.read_csv("data/ECM_simulation_OCV_ocv7.csv",
                               names=['Test_Time(s)', 'Current(A)', 'Voltage(V)', 'SoC', 'Vc'])
        elif j == 1:
            data = pd.read_csv("data/ECM_simulation_FUDS_ocv7.csv",
                               names=['Test_Time(s)', 'Current(A)', 'Voltage(V)', 'SoC', 'Vc'])
        elif j == 2:
            data = pd.read_csv("data/ECM_simulation_2CDist_ocv7.csv",
                               names=['Test_Time(s)', 'Current(A)', 'Voltage(V)', 'SoC', 'Vc'])
        elif j == 3:
            data = pd.read_csv("data/ECM_simulation_US06_ocv7.csv",
                               names=['Test_Time(s)', 'Current(A)', 'Voltage(V)', 'SoC', 'Vc'])
        elif j == 4:
            data = pd.read_csv("data/ECM_simulation_DST_ocv7.csv",
                               names=['Test_Time(s)', 'Current(A)', 'Voltage(V)', 'SoC', 'Vc'])
        elif j == 5:
            data = pd.read_csv("data/ECM_simulation_BJDST_ocv7.csv",
                               names=['Test_Time(s)', 'Current(A)', 'Voltage(V)', 'SoC', 'Vc'])
        else:
            data = pd.read_csv("data/SPM_Pade3rd_simulation_UDDSx2.csv",
                               names=['Test_Time(s)', 'Current(A)', 'Voltage(V)',
                                      'Css_n', 'Cs_ave_n', 'Css_p', 'Cs_ave_p'])
        df = pd.concat([df, data], ignore_index=True)
        length.append(len(data))
    return df, length


def load_data_pade(data_type, pade_order):
    df = pd.DataFrame()
    length = []
    print()
    for j in data_type:
        if j == 1:
            if pade_order == 2:
                data = pd.read_csv("data/SPM_Pade2nd_simulation_UDDSx2.csv",
                                   names=['Test_Time(s)', 'Current(A)', 'Voltage(V)',
                                          'Css_n', 'Cs_ave_n', 'Css_p', 'Cs_ave_p',
                                          'Xn_1', 'Xn_2', 'Xp_1', 'Xp_2'])
            elif pade_order == 3:
                data = pd.read_csv("data/SPM_Pade3rd_simulation_UDDSx2.csv",
                                   names=['Test_Time(s)', 'Current(A)', 'Voltage(V)',
                                          'Css_n', 'Cs_ave_n', 'Css_p', 'Cs_ave_p',
                                          'Xn_1', 'Xn_2', 'Xn_3', 'Xp_1', 'Xp_2', 'Xp_3'])
            elif pade_order == 4:
                data = pd.read_csv("data/SPM_Pade4th_simulation_UDDSx2.csv",
                                   names=['Test_Time(s)', 'Current(A)', 'Voltage(V)',
                                          'Css_n', 'Cs_ave_n', 'Css_p', 'Cs_ave_p',
                                          'Xn_1', 'Xn_2', 'Xn_3', 'Xn_4', 'Xp_1', 'Xp_2', 'Xp_3', 'Xp_4'])
            else:
                raise Exception("Pade order must be 2, 3, or 4")
        else:
            raise Exception("Data type must be 1")
        df = pd.concat([df, data], ignore_index=True)
        length.append(len(data))
    return df, length


def load_data_spmfdm(data_type, states=False, cell=1):
    df = pd.DataFrame()
    length = []
    if states:
        for j in data_type:
            if j == 1:
                data = pd.read_csv("data/SPM_FDM_nr20_simulation_UDDSx2_cell"+str(cell)+".csv", header=None)
            elif j == 2:
                data = pd.read_csv("data/SPM_FDM_nr20_simulation_FUDS_cell"+str(cell)+".csv", header=None)
            elif j == 3:
                data = pd.read_csv("data/SPM_FDM_nr20_simulation_US06_Extended_cell"+str(cell)+".csv", header=None)
            elif j == 4:
                data = pd.read_csv("data/SPM_FDM_nr20_simulation_Charge_cell"+str(cell)+".csv", header=None)
            elif j == 5:
                data = pd.read_csv("data/SPM_FDM_nr20_simulation_Charge2_cell" + str(cell) + ".csv", header=None)
            else:
                raise Exception("Data type must be within 1-4")
            df = pd.concat([df, data], ignore_index=True)
            length.append(len(data))
    else:
        for j in data_type:
            if j == 1:
                data = pd.read_csv("data/SPM_FDM_nr100_simulation_UDDSx2_cell"+str(cell)+".csv",
                                   names=['Test_Time(s)', 'Current(A)', 'Voltage(V)',
                                          'Css_n', 'Cs_ave_n', 'Css_p', 'Cs_ave_p'])
            elif j == 2:
                data = pd.read_csv("data/SPM_FDM_nr100_simulation_FUDS_cell"+str(cell)+".csv",
                                   names=['Test_Time(s)', 'Current(A)', 'Voltage(V)',
                                          'Css_n', 'Cs_ave_n', 'Css_p', 'Cs_ave_p'])
            elif j == 3:
                data = pd.read_csv("data/SPM_FDM_nr100_simulation_US06_Extended_cell"+str(cell)+".csv",
                                   names=['Test_Time(s)', 'Current(A)', 'Voltage(V)',
                                          'Css_n', 'Cs_ave_n', 'Css_p', 'Cs_ave_p'])
            elif j == 4:
                data = pd.read_csv("data/SPM_FDM_nr100_simulation_Charge_cell"+str(cell)+".csv",
                                   names=['Test_Time(s)', 'Current(A)', 'Voltage(V)',
                                          'Css_n', 'Cs_ave_n', 'Css_p', 'Cs_ave_p'])
            elif j == 5:
                data = pd.read_csv("data/SPM_FDM_nr100_simulation_Charge2_cell" + str(cell) + ".csv",
                                   names=['Test_Time(s)', 'Current(A)', 'Voltage(V)',
                                          'Css_n', 'Cs_ave_n', 'Css_p', 'Cs_ave_p'])
            else:
                raise Exception("Data type must be within 1-4")
            df = pd.concat([df, data], ignore_index=True)
            length.append(len(data))
    return df, length


def create_sequential(x, k, data_length, head=0, tail=0):
    x_seq_all = 0
    index = 0
    for n in data_length:
        x_seq = x[index + head:index + n - k + 1 - tail]
        for j in range(k - 1):
            x_seq = np.vstack((x_seq, x[index + j + 1 + head:index + n - k + 1 + j + 1 - tail]))
        if index == 0:
            x_seq_all = x_seq
        else:
            x_seq_all = np.hstack((x_seq_all, x_seq))
        index += n
    return x_seq_all.T


def calc_ave(x_seq, h):
    x_seq_ave = torch.zeros(x_seq.shape[0], int(x_seq.shape[1] / h))
    for i in range(x_seq_ave.shape[1]):
        x_seq_ave[:, i] = torch.sum(x_seq[:, i*h:(i+1)*h], dim=1) / h
    return x_seq_ave


def sampling(x_seq, h):
    x_seq_sampled = torch.zeros(x_seq.shape[0], int(x_seq.shape[1] / h))
    for i in range(int(x_seq.shape[1] / h)):
        x_seq_sampled[:, i] = x_seq[:, (i + 1) * h - 1]
    return x_seq_sampled


def create_averaged_inputs(x, r, data_length, k=0):
    x_seq = create_sequential(x, 2 ** r, data_length, 0, k)
    n = len(x_seq)
    x_ave_seq_all = x[0:n]
    for j in range(r):
        x_ave_seq = np.average(x_seq[:, 2 ** r - 2 ** (j + 1):], 1)
        x_ave_seq_all = np.vstack((x_ave_seq_all, x_ave_seq))
    return x_ave_seq_all.T


def time_derivative_auto(out_pred, inputs, h):
    dx_dt = torch.zeros((len(inputs) - 2, out_pred.shape[1]))
    for i in range(out_pred.shape[1]):
        dx_du = gradients(out_pred[i:i + 1], inputs)[0]
        for j in range(inputs.shape[1]):
            u = inputs[:, j:j + 1]
            du_dt = euler(u) / h
            dx_du_j = dx_du[1:-1, j:j + 1]
            dx_dt[:, j:j + 1] = dx_du_j * du_dt
    return dx_dt


def gradients(outputs, inputs):
    return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True,
                               allow_unused=True)


def euler(x):
    #    x_ahead = torch.cat((x[1:, ], x[-1, ].unsqueeze(-1)))
    x_ahead = x[2:, ]
    #    x_prev = torch.cat((torch.tensor([0]).unsqueeze(-1), x[:-1, ]))
    x_prev = x[:-2, ]
    return (x_ahead - x_prev) / 2

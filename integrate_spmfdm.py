# ref:https://qiita.com/windfall/items/72b303867b174875f1b3
import torch.nn as nn
import torch
import integrator as integrator
import spm_fdm


class IntegrateSPM(nn.Module):
    def __init__(self, p, cfg):
        super().__init__()
        self.spm_model = spm_fdm.SPMFdm(p, cfg['n_r'])
        self.n_r = cfg['n_r']
        self.p = p
        self.delta_r_bar = 1 / self.n_r
        self.R_f_n_init = p.R_f_n
        self.k_n_init = p.k_n
        self.k_p_init = p.k_p
        self.D_s_n_init = p.D_s_n
        self.D_s_p_init = p.D_s_p
        self.nLi_s_init = p.nLi_s

        if cfg['p_targets'][0]:
            self.nLi_s_scaled = torch.tensor([p.nLi_s / self.nLi_s_init], requires_grad=True).float()
            self.nLi_s_scaled = nn.Parameter(self.nLi_s_scaled)
        else:
            self.nLi_s_scaled = torch.tensor([p.nLi_s_true / self.nLi_s_init], requires_grad=False).float()

        if cfg['p_targets'][1]:
            self.R_f_n_scaled = torch.tensor([p.R_f_n / self.R_f_n_init], requires_grad=True).float()
            self.R_f_n_scaled = nn.Parameter(self.R_f_n_scaled)
        else:
            self.R_f_n_scaled = torch.tensor([p.R_f_n_true / self.R_f_n_init], requires_grad=False).float()

        if cfg['p_targets'][2]:
            self.k_n_scaled = torch.tensor([p.k_n / self.k_n_init], requires_grad=True).float()
            self.k_n_scaled = nn.Parameter(self.k_n_scaled)
        else:
            self.k_n_scaled = torch.tensor([p.k_n_true / self.k_n_init], requires_grad=False).float()

        if cfg['p_targets'][3]:
            self.k_p_scaled = torch.tensor([p.k_p / self.k_p_init], requires_grad=True).float()
            self.k_p_scaled = nn.Parameter(self.k_p_scaled)
        else:
            self.k_p_scaled = torch.tensor([p.k_p_true / self.k_p_init], requires_grad=False).float()

        if cfg['p_targets'][4]:
            self.D_s_n_scaled = torch.tensor([p.D_s_n / self.D_s_n_init], requires_grad=True).float()
            self.D_s_n_scaled = nn.Parameter(self.D_s_n_scaled)
        else:
            self.D_s_n_scaled = torch.tensor([p.D_s_n_true / self.D_s_n_init], requires_grad=False).float()

        if cfg['p_targets'][5]:
            self.D_s_p_scaled = torch.tensor([p.D_s_p / self.D_s_p_init], requires_grad=True).float()
            self.D_s_p_scaled = nn.Parameter(self.D_s_p_scaled)
        else:
            self.D_s_p_scaled = torch.tensor([p.D_s_p_true / self.D_s_p_init], requires_grad=False).float()

        self.alpha_n = self.alpha(p.D_s_n, p.R_s_n)
        self.alpha_p = self.alpha(p.D_s_p, p.R_s_p)
        self.beta_n = - self.beta(p.R_s_n, p.D_s_n, p.a_s_n, p.A_n, p.L_n)
        self.beta_p = self.beta(p.R_s_p, p.D_s_p, p.a_s_p, p.A_p, p.L_p)
        self.matAn = self.alpha_n * self.spm_model.matA
        self.matBn = self.alpha_n * self.beta_n * self.spm_model.matB
        self.matAp = self.alpha_p * self.spm_model.matA
        self.matBp = self.alpha_p * self.beta_p * self.spm_model.matB
        self.h = cfg['h']  # step size for integrator

        'Setting up integration layer'
        if cfg['integ_type'] == 0:
            self.integration = integrator.RK4(self.spm_model, h=self.h)
        else:
            self.integration = integrator.Naive(self.spm_model, h=self.h)

    def forward(self, x, i_seq, k):
        x_all = torch.zeros(x.shape[0], x.shape[1], k)
        xn = x[:, :self.n_r - 1]
        xp = x[:, self.n_r - 1:]

        for j in range(k):
            xn = self.integration(xn, i_seq[:, j:j + 1], torch.tensor([0]).float(), self.matAn, self.matBn)
            xp = self.integration(xp, i_seq[:, j:j + 1], torch.tensor([0]).float(), self.matAp, self.matBp)
            x_all[:, :self.n_r - 1, j] = xn
            x_all[:, self.n_r - 1:, j] = xp
        return x_all

    def alpha(self, D_s, R_s):
        return D_s / ((R_s * self.delta_r_bar) ** 2)

    def beta(self, R_s, D_s, a_s, A, L):
        return 2 * self.delta_r_bar * R_s / (D_s * self.p.Faraday * a_s * A * L)

    def update_dynamics(self):
        self.p.D_s_n = self.D_s_n_scaled * self.D_s_n_init
        self.p.D_s_p = self.D_s_p_scaled * self.D_s_p_init
        self.p.R_f_n = self.R_f_n_scaled * self.R_f_n_init
        self.p.k_n = self.k_n_scaled * self.k_n_init
        self.p.k_p = self.k_p_scaled * self.k_p_init
        self.p.nLi_s = self.nLi_s_scaled * self.nLi_s_init
        self.alpha_n = self.alpha(self.p.D_s_n, self.p.R_s_n)
        self.alpha_p = self.alpha(self.p.D_s_p, self.p.R_s_p)
        self.beta_n = - self.beta(self.p.R_s_n, self.p.D_s_n, self.p.a_s_n, self.p.A_n, self.p.L_n)
        self.beta_p = - self.beta(self.p.R_s_p, self.p.D_s_p, self.p.a_s_p, self.p.A_p, self.p.L_p)
        self.matAn = self.alpha_n * self.spm_model.matA
        self.matBn = self.alpha_n * self.beta_n * self.spm_model.matB

    def calc_css_and_cs0(self, x, i):
        'Calculating central and surface concentration from NN prediction'
        matC = self.spm_model.matC
        matDn = self.beta_n * self.spm_model.matD
        matDp = self.beta_p * self.spm_model.matD
        cs_n = torch.zeros(len(i), self.n_r + 1)
        cs_p = torch.zeros(len(i), self.n_r + 1)

        # using BC for calculating central and surface concentration and store all the states
        cs0_n = matC[0, 0] * x[:, 0] + matC[0, 1] * x[:, 1]
        cs0_p = matC[0, 0] * x[:, self.n_r - 1] + matC[0, 1] * x[:, self.n_r]
        css_n = matC[1, self.n_r - 3] * x[:, self.n_r - 3] \
                + matC[1, self.n_r - 2] * x[:, self.n_r - 2] \
                + matDn[1] * i
        css_p = matC[1, self.n_r - 3] * x[:, -2] \
                + matC[1, self.n_r - 2] * x[:, -1] \
                + matDp[1] * i
        cs_n[:, 0] = cs0_n
        cs_n[:, 1:-1] = x[:, :self.n_r - 1]
        cs_n[:, -1] = css_n
        cs_p[:, 0] = cs0_p
        cs_p[:, 1:-1] = x[:, self.n_r - 1:]
        cs_p[:, -1] = css_p
        return cs_n, cs_p

    def calc_css_and_cs0_seq(self, x, i):
        'Calculating central and surface concentration from NN prediction'
        matC = self.spm_model.matC
        matDn = self.beta_n * self.spm_model.matD
        matDp = self.beta_p * self.spm_model.matD
        cs_n = torch.zeros(i.shape[0], self.n_r + 1, i.shape[1])
        cs_p = torch.zeros(i.shape[0], self.n_r + 1, i.shape[1])

        # using BC for calculating central and surface concentration and store all the states
        cs0_n = matC[0, 0] * x[:, 0, :] + matC[0, 1] * x[:, 1, :]
        cs0_p = matC[0, 0] * x[:, self.n_r - 1, :] + matC[0, 1] * x[:, self.n_r]
        css_n = matC[1, self.n_r - 3] * x[:, self.n_r - 3, :] \
                + matC[1, self.n_r - 2] * x[:, self.n_r - 2, :] \
                + matDn[1] * i
        css_p = matC[1, self.n_r - 3] * x[:, -2, :] \
                + matC[1, self.n_r - 2] * x[:, -1, :] \
                + matDp[1] * i
        cs_n[:, 0, :] = cs0_n
        cs_n[:, 1:-1, :] = x[:, :self.n_r - 1, :]
        cs_n[:, -1, :] = css_n
        cs_p[:, 0, :] = cs0_p
        cs_p[:, 1:-1, :] = x[:, self.n_r - 1:, :]
        cs_p[:, -1, :] = css_p
        return cs_n, cs_p
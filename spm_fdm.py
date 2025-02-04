# ref:https://github.com/rtqichen/torchdiffeq/issues/128
import torch
import torch.nn as nn
import math


def ref_potential_anode(theta):
    # Polynomial Fit
    # DUALFOIL: MCMB 2528 graphite (Bellcore) 0.01 < x < 0.9
    Uref = 0.194 + 1.5 * torch.exp(-120.0 * theta) \
           + 0.0351 * torch.tanh((theta - 0.286) / 0.083) \
           - 0.0045 * torch.tanh((theta - 0.849) / 0.119) \
           - 0.035 * torch.tanh((theta - 0.9233) / 0.05) \
           - 0.0147 * torch.tanh((theta - 0.5) / 0.034) \
           - 0.102 * torch.tanh((theta - 0.194) / 0.142) \
           - 0.022 * torch.tanh((theta - 0.9) / 0.0164) \
           - 0.011 * torch.tanh((theta - 0.124) / 0.0226) \
           + 0.0155 * torch.tanh((theta - 0.105) / 0.029)
    return Uref


def ref_potential_cathode(theta):
    # Polynomial Fit
    # DUALFOIL: CoO2 (Cobalt Dioxide) 0.5 < y < 0.99
    Uref = 2.16216 + 0.07645 * torch.tanh(30.834 - 54.4806 * theta) \
           + 2.1581 * torch.tanh(52.294 - 50.294 * theta) \
           - 0.14169 * torch.tanh(11.0923 - 19.8543 * theta) \
           + 0.2051 * torch.tanh(1.4684 - 5.4888 * theta) \
           + 0.2531 * torch.tanh((-theta + 0.56478) / 0.1316) \
           - 0.02167 * torch.tanh((theta - 0.525) / 0.006)
    return Uref


class SPMFdm(nn.Module):
    def __init__(self, p, n_r):
        super().__init__()
        self.p = p
        self.n_r = n_r
        self.in_features = self.n_r  # [c_1 ... c_Nr-1, i]
        self.out_features = self.n_r - 1  # [c_1 ... c_Nr-1]
        self.RTaF = (p.R * p.T_amb) / (p.alph * p.Faraday)
        self.matA = self.calc_matA()
        self.matB = self.calc_matB()
        self.matC = self.calc_matC()
        self.matD = self.calc_matD()

    def forward(self, state, matA, matB):
        # state = [c_1 ... c_Nr-1, I]
        dxdt = state[:, 0:self.n_r - 1] @ matA.T + state[:, self.n_r - 1:self.n_r] @ matB.T
        return dxdt

    def calc_matA(self):
        return self.matM1() - self.matM2() @ (torch.inverse(self.matN2())) @ self.matN1()

    def calc_matB(self):
        return self.matM2() @ torch.inverse(self.matN2()) @ self.matN3()

    def calc_matC(self):
        return - torch.inverse(self.matN2()) @ self.matN1()

    def calc_matD(self):
        return torch.inverse(self.matN2()) @ self.matN3()

    def matM1(self):
        matM1 = torch.zeros(self.n_r - 1, self.n_r - 1)
        for i in range(self.n_r - 1):
            matM1[i, i] = -2.0
            if i >= 1:
                matM1[i, i - 1] = i / (i + 1)
            if i <= (self.n_r - 3):
                matM1[i, i + 1] = (i + 2) / (i + 1)
        return matM1

    def matM2(self):
        matM2 = torch.zeros(self.n_r - 1, 2)
        matM2[self.n_r - 2, 1] = self.n_r / (self.n_r - 1)
        return matM2

    def matN1(self):
        matN1 = torch.zeros(2, self.n_r - 1)
        matN1[0, 0] = 4.0
        matN1[0, 1] = -1.0
        matN1[1, self.n_r - 3] = 1.0
        matN1[1, self.n_r - 2] = -4.0
        return matN1

    def matN2(self):
        return torch.tensor([[-3.0, 0.0], [0.0, 3.0]])

    def matN3(self):
        return torch.tensor([[0.0], [1.0]])

    def calc_voltage(self, css_n, css_p, ce, I, k_n, k_p, R_f_n):
        css_n[css_n > self.p.c_s_n_max * 0.999] = self.p.c_s_n_max * 0.999
        css_p[css_p > self.p.c_s_p_max * 0.999] = self.p.c_s_p_max * 0.999
        css_n[css_n < 1] = 1
        css_p[css_p < 1] = 1

        # Stochiometric Concentration Ratio
        theta_n = css_n / self.p.c_s_n_max
        theta_p = css_p / self.p.c_s_p_max

        # Equilibrium Potential
        Unref = ref_potential_anode(theta_n)
        Upref = ref_potential_cathode(theta_p)

        # Exchange Current Density
        i_0n = self.exch_cur_dens(css_n, ce, k_n, self.p.c_s_n_max)
        i_0p = self.exch_cur_dens(css_p, ce, k_p, self.p.c_s_p_max)

        V = self.RTaF * torch.arcsinh(-I / (2 * self.p.a_s_p * self.p.L_p * i_0p)) \
            - self.RTaF * torch.arcsinh(I / (2 * self.p.a_s_n * self.p.L_n * i_0n)) \
            + Upref - Unref \
            - (R_f_n / (self.p.a_s_n * self.p.L_n) + self.p.R_f_p / (self.p.a_s_p * self.p.L_p)) * I

        return V

    def exch_cur_dens(self, css, ce, k, cs_max):
        return k * (torch.mul((cs_max - css), css) * ce) ** self.p.alph

    def calc_cs_bar_seq(self, cs_seq):
        c_ave = torch.zeros(cs_seq.shape[0], cs_seq.shape[2])
        for i in range(self.n_r):
            r_bar_upper = (i + 1) / self.n_r
            r_bar_lower = i / self.n_r
            c_ave += 4 * math.pi / 3 * (r_bar_upper ** 3 - r_bar_lower ** 3) \
                     * (1 / 2) * (cs_seq[:, i, :] + cs_seq[:, i + 1, :])

        return c_ave / (4 / 3 * math.pi)

    def calc_cs_bar(self, cs):
        c_ave = torch.zeros(cs.shape[0])
        for i in range(self.n_r):
            r_bar_upper = (i + 1) / self.n_r
            r_bar_lower = i / self.n_r
            c_ave += 4 * math.pi / 3 * (r_bar_upper ** 3 - r_bar_lower ** 3) \
                     * (1 / 2) * (cs[:, i] + cs[:, i + 1])

        return c_ave / (4 / 3 * math.pi)

    def init_cs(self, V):

        # Algorithm params
        maxiters = 50
        x = torch.zeros(maxiters, 1)
        f = torch.zeros(maxiters, 1)
        tol = 1e-5

        # Initial Guesses
        x_low = 0.2 * self.p.c_s_p_max
        x_high = 1.0 * self.p.c_s_p_max
        x[0] = 0.6 * self.p.c_s_p_max

        # Iterate Bisection Algorithm
        final_idx = 0
        for idx in range(maxiters - 1):
            theta_p = x[idx] / self.p.c_s_p_max
            theta_n = (self.p.n_Li_s - self.p.epsilon_s_p * self.p.L_p * self.p.A_p * x[idx]) \
                      / (self.p.c_s_n_max * self.p.epsilon_s_n * self.p.L_n * self.p.A_n)

            OCPn = ref_potential_anode(theta_n)
            OCPp = ref_potential_cathode(theta_p)

            f[idx] = OCPp - OCPn - V

            if abs(f[idx]) <= tol:
                break
            elif f[idx] <= 0:
                x_high = x[idx]
            else:
                x_low = x[idx]

            # Bisection
            x[idx + 1] = (x_high + x_low) / 2
            final_idx += 1

        # Output converged csp0
        csp0 = x[final_idx]
        # Compute csn0
        csn0 = (self.p.n_Li_s - self.p.epsilon_s_p * self.p.L_p * self.p.A_p * csp0) \
               / (self.p.epsilon_s_n * self.p.L_n * self.p.A_n)
        return torch.tensor([csn0, csp0])

class InitParams:
    def __init__(self, cfg):
        'Battery parameters'
        # Geometric params
        self.L_n = 100e-6  # Thickness of negative electrode [m]
        self.L_p = 100e-6  # Thickness of positive electrode [m]
        self.A_n = 1  # Area of negative electrode [m2]
        self.A_p = 1  # Area of negative electrode [m2]
        self.R_s_n = 10e-6  # Radius of solid particles in negative electrode [m]
        self.R_s_p = 10e-6  # Radius of solid particles in positive electrode [m]
        self.epsilon_s_n = 0.6  # Volume fraction in solid for neg.electrode
        self.epsilon_s_p = 0.5  # Volume fraction in solid for pos.electrode
        self.density_n = 1800  # Mass density of anode material [kg/m3]
        self.density_p = 5010  # Mass density of anode material [kg/m3]
        self.a_s_n = 3 * self.epsilon_s_n / self.R_s_n  # Specific surface area for neg.electrode [m^2/m^3]
        self.a_s_p = 3 * self.epsilon_s_p / self.R_s_p  # Specific surface area for pos.electrode [m^2/m^3]

        # Transport Params
        self.D_s_n = 3.9e-14  # Diffusion coeff for solid in neg. electrode, [m^2/s]
        self.D_s_p = 1e-13  # Diffusion coeff for solid in pos. electrode, [m^2/s]
        self.Faraday = 96487  # Faraday's constant, [Coulumbs/mol]

        # Kinetic params
        self.R = 8.314472  # Gas constant [J/mol-K]
        self.alph = 0.5  # Charge transfer coefficients
        self.R_f_n = 1e-3  # Resistivity of SEI layer, [Ohms * m ^ 2]
        self.R_f_p = 0  # Resistivity of SEI layer, [Ohms * m ^ 2]
        # Reaction rate constant[m2.5mol - 0.5s - 1]*[C / mol] = [(A / m ^ 2) * (m ^ 3 / mol) ^ (1.5)] from
        # https: // github.com / davidhowey / Spectral_li - ion_SPM / blob / master / model_parameters / get_modelData.m
        self.k_n = 1.764e-11 * self.Faraday
        self.k_p = 6.667e-11 * self.Faraday

        # Thermodynamics params
        self.T_amb = 298  # Ambient temperature in [K]

        # Concentrations
        self.specific_capa_n = 372  # Anode material's specific capacity [mAh/g]
        self.specific_capa_p = 247  # Cathode material's specific capacity [mAh/g]
        self.c_s_n_max = 3.6e3 * self.specific_capa_n * self.density_n \
                         / self.Faraday  # Max concentration in anode, [mol / m ^ 3]
        self.c_s_p_max = 3.6e3 * self.specific_capa_p * self.density_p \
                         / self.Faraday  # Max concentration in cathode, [mol / m ^ 3]
        self.nLi_s = 2.50  # Total moles of lithium in solid phase[mol]
        self.c_e = 1e3  # Fixed electrolyte concentration for SPM, [mol / m ^ 3]

        # Set true value of the parameter to be estimated
        if cfg['cell_target'] == 1:  # fresh cell
            self.nLi_s_true = self.nLi_s
            self.R_f_n_true = self.R_f_n
            self.k_n_true = self.k_n
            self.k_p_true = self.k_p
            self.D_s_n_true = self.D_s_n
            self.D_s_p_true = self.D_s_p
        else:  # degraded cell
            self.nLi_s_true = self.nLi_s * 0.8
            self.R_f_n_true = self.R_f_n * 40
            self.k_n_true = self.k_n
            self.k_p_true = self.k_p
            self.D_s_n_true = self.D_s_n
            self.D_s_p_true = self.D_s_p

        # Correct parameter if pre-trained NN is trained on degraded cell
        if cfg['cell_known'] == 2:  # degraded cell
            self.nLi_s = self.nLi_s * 0.8
            self.R_f_n = self.R_f_n * 40
            self.k_n = self.k_n
            self.k_p = self.k_p
            self.D_s_n = self.D_s_n
            self.D_s_p = self.D_s_p

        # Set true value if the parameter is not to be estimated
        if not cfg['p_targets'][0]:
            self.nLi_s = self.nLi_s_true
        if not cfg['p_targets'][1]:
            self.R_f_n = self.R_f_n_true
        if not cfg['p_targets'][2]:
            self.k_n = self.k_n_true
        if not cfg['p_targets'][3]:
            self.k_p = self.k_p_true
        if not cfg['p_targets'][4]:
            self.D_s_n = self.D_s_n_true
        if not cfg['p_targets'][5]:
            self.D_s_p = self.D_s_p_true


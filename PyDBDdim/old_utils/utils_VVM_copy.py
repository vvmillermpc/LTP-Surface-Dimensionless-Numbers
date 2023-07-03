import numpy as np
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import PolynomialFeatures
from .BOutils import InnerBO, OuterBO, MixedBO
from sympy import Matrix
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import os
import torch
import scipy
import matplotlib.pyplot as plt
import platform

device = "cpu" if torch.cuda.is_available() else "cpu"

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cpu" if torch.cuda.is_available() else "cpu"),
}
gen_kwargs = {
    "device": torch.device("cpu" if torch.cuda.is_available() else "cpu"),
}


class DimensionlessLearning(object):
    def __init__(self, input_dimension, output_dimension):
        self.input_dimension = torch.tensor(input_dimension, **tkwargs)
        # (num_basic_unit, num_variables)
        self.output_dimension = torch.tensor(output_dimension, **tkwargs)
        # (num_basic_unit, 1)
        self._basis_col = None
        self.X = None
        self.y = None
        self.best_mse = float('-inf')
        self.best_inner_coef = None

    @property
    def basis_col(self):
        sympy_matrix = Matrix(self.input_dimension)
        self._basis_col = torch.tensor(sympy_matrix.nullspace(), **tkwargs).squeeze(-1).transpose(0, 1)
        # (num_variables, num_basis)
        return self._basis_col

    def read_data(self, X, y, random_state=0):
        self.X = torch.tensor(X, **tkwargs)
        self.y = torch.tensor(y, **tkwargs).reshape(-1, 1)
        shuffle(self.X, self.y, random_state=random_state)

    def inject_coef(self, coef):
        coef_pi = torch.matmul(self.basis_col, coef.reshape(-1, 1)).reshape(1, -1)

        def predict(X, coef_w, degree):
            pi_in = torch.prod(torch.pow(X, coef_pi), dim=1).reshape(1, -1)
            # (1, batch_size)
            feats_list = []
            for degree_idx in range(degree+1):
                feats_list.append(pi_in**degree_idx)
            feats = torch.vstack(feats_list).transpose(0, 1)
            # (batch_size, degree+1)
            pred = torch.matmul(feats, coef_w).reshape(-1, 1)
            return pred
        return predict

    def metric(self, coef):
        coef_pi = torch.matmul(self.basis_col, coef.reshape(-1, 1)).reshape(1, -1)

        def eval_nfold(alpha_and_degree):
            alpha_and_degree = alpha_and_degree.reshape(-1)
            # print(alpha_and_degree)
            assert alpha_and_degree.shape[0] == 2
            alpha = 10**(alpha_and_degree[0].item())
            degree = int(alpha_and_degree[1].item())
            ss = KFold(n_splits=5, random_state=0, shuffle=True)
            pi_in = torch.prod(torch.pow(self.X, coef_pi), dim=1).reshape(1, -1)
            # (1, batch_size)
            feats_list = []
            for degree_idx in range(degree+1):
                feats_list.append(pi_in**degree_idx)
            feats = torch.vstack(feats_list).transpose(0, 1)
            # (batch_size, degree+1)
            score = 0
            for seed, (train_index, val_index) in enumerate(ss.split(feats)):
                train_X, val_X = feats[train_index], feats[val_index]
                train_y, val_y = self.y[train_index], self.y[val_index]

                model = Lasso(alpha=alpha, fit_intercept=False)
                model.fit(train_X, train_y)
                pred = model.predict(val_X)
                score += torch.sum((val_y-pred)**2)
            return -score/self.X.shape[0]
            # Botorch assumes maximization, so we return the negative score
        return eval_nfold

    def metric_mixed_poly(self, para, eval_method='R2'):
        para = para.reshape(-1)
        print(para)
        n_dimensionless = (para.shape[0]-2)//self.basis_col.shape[1]
        assert para.shape[0] == self.basis_col.shape[1]*n_dimensionless+2
        alpha = 10**(para[-2].item())
        degree = int(para[-1].item())
        pi_in = torch.empty((self.X.shape[0], 0), **tkwargs)
        for i in range(n_dimensionless):
            coef = para[i*self.basis_col.shape[1]:(i+1)*self.basis_col.shape[1]]
            coef_pi = torch.matmul(self.basis_col, coef.reshape(-1, 1)).reshape(1, -1)
            pi_in = torch.hstack((pi_in, torch.prod(torch.pow(self.X, coef_pi), dim=1).reshape(-1, 1)))
        print(pi_in.shape)
        # (batch_size, n_dimensionless)

        ss = KFold(n_splits=5, random_state=0, shuffle=True)
        poly = PolynomialFeatures(degree)

        score = 0
        for seed, (train_index, val_index) in enumerate(ss.split(pi_in)):
            train_X, val_X = pi_in[train_index], pi_in[val_index]
            train_y, val_y = self.y[train_index], self.y[val_index]

            poly_train_X = poly.fit_transform(train_X)
            poly_val_X = poly.fit_transform(val_X)

            model = Lasso(alpha=alpha, fit_intercept=False)
            model.fit(poly_train_X, train_y)
            pred = model.predict(poly_val_X)
            if eval_method == 'mse':
                score += torch.sum((val_y-pred)**2)
            elif eval_method == 'R2':
                score += torch.tensor(r2_score(val_y, pred)*val_X.shape[0])

        return -score/self.X.shape[0] if eval_method == 'mse' else score/self.X.shape[0]

    def mixed_poly_loop_res(self, bounds, N_BATCH=100, NUM_RESTARTS=10, RAW_SAMPLES=512, INITIAL_SIZE=10, N_TRIAL = 1, verbose=False):
        if isinstance(bounds, torch.Tensor):
            bounds = bounds.clone().to(**tkwargs)
        else:
            bounds = torch.tensor(bounds, **tkwargs)
        x_dim = bounds.shape[1]
        x_cat_dim = [bounds.shape[1]-1]
        MixedBO_instance = MixedBO(self.metric_mixed_poly, NUM_RESTARTS, N_TRIAL, N_BATCH, RAW_SAMPLES, INITIAL_SIZE, x_dim, x_cat_dim, bounds)
        best_observed_y, best_observed_x, _ = MixedBO_instance.train(verbose=verbose)
        if best_observed_y > self.best_mse:
            self.best_mse = best_observed_y
            self.best_total_coef = best_observed_x
        return torch.tensor([best_observed_y], **tkwargs).reshape(-1, 1)
    def metric_mixed(self, para, eval_method='R2'):
        para = para.reshape(-1)
        print(para)
        assert para.shape[0] == self.basis_col.shape[1]+2
        coef = para[:self.basis_col.shape[1]]
        alpha = 10**(para[-2].item())
        degree = int(para[-1].item())
        coef_pi = torch.matmul(self.basis_col, coef.reshape(-1, 1)).reshape(1, -1)
        ss = KFold(n_splits=5, random_state=0, shuffle=True)
        pi_in = torch.prod(torch.pow(self.X, coef_pi), dim=1).reshape(1, -1)
        # (1, batch_size)
        feats_list = []
        for degree_idx in range(degree+1):
            feats_list.append(pi_in**degree_idx)
        feats = torch.vstack(feats_list).transpose(0, 1)
        # (batch_size, degree+1)
        score = 0
        for seed, (train_index, val_index) in enumerate(ss.split(feats)):
            train_X, val_X = feats[train_index], feats[val_index]
            train_y, val_y = self.y[train_index], self.y[val_index]

            model = Lasso(alpha=alpha, fit_intercept=False)
            model.fit(train_X, train_y)
            pred = model.predict(val_X)
            if eval_method == 'mse':
                score += torch.sum((val_y-pred)**2)
            elif eval_method == 'R2':
                score += torch.tensor(r2_score(val_y, pred)*val_X.shape[0])

        return -score/self.X.shape[0] if eval_method == 'mse' else score/self.X.shape[0]

    def mixed_loop_res(self, bounds, N_BATCH=100, NUM_RESTARTS=10, RAW_SAMPLES=512, INITIAL_SIZE=10, N_TRIAL = 1, verbose=False):
        # bounds1 = torch.tensor([[-1, 1]]*self.basis_col.shape[1], **tkwargs).transpose(0, 1)
        # bounds2 = torch.tensor([[-8, 1], [1, 6]], **tkwargs)
        # bounds = torch.cat([bounds1, bounds2], dim=1)
        # Log transform
        if isinstance(bounds, torch.Tensor):
            bounds = bounds.clone().to(**tkwargs)
        else:
            bounds = torch.tensor(bounds, **tkwargs)
        x_dim = self.basis_col.shape[1]+2
        x_cat_dim = [self.basis_col.shape[1]+1]
        MixedBO_instance = MixedBO(self.metric_mixed, NUM_RESTARTS, N_TRIAL, N_BATCH, RAW_SAMPLES, INITIAL_SIZE, x_dim, x_cat_dim, bounds)
        best_observed_y, best_observed_x, _ = MixedBO_instance.train(verbose=verbose)
        if best_observed_y > self.best_mse:
            self.best_mse = best_observed_y
            self.best_total_coef = best_observed_x
        return torch.tensor([best_observed_y], **tkwargs).reshape(-1, 1)

    def inner_loop_res(self, coef):
        model = self.metric(coef)
        bounds = torch.tensor([[-8, 1], [1, 8]], **tkwargs)
        # Log transform
        N_BATCH = 100
        NUM_RESTARTS = 10
        RAW_SAMPLES = 512
        INITIAL_SIZE = 10
        N_TRIAL = 1
        x_dim = 2
        x_cat_dim = [1]
        InnerBO_instance = InnerBO(model, NUM_RESTARTS, N_TRIAL, N_BATCH, RAW_SAMPLES, INITIAL_SIZE, x_dim, x_cat_dim, bounds)
        best_observed_y, best_observed_x, _ = InnerBO_instance.train()
        if best_observed_y > self.best_mse:
            self.best_mse = best_observed_y
            self.best_inner_coef = best_observed_x
        return torch.tensor([best_observed_y], **tkwargs).reshape(-1, 1)

    def outer_loop_res(self, verbose=False):
        bounds = torch.tensor([[-1, 1]]*self.basis_col.shape[1], **tkwargs).transpose(0, 1)
        N_BATCH = 100
        NUM_RESTARTS = 10
        RAW_SAMPLES = 512
        INITIAL_SIZE = 10
        N_TRIAL = 1
        x_dim = self.basis_col.shape[1]
        OuterBO_instance = OuterBO(self.inner_loop_res, NUM_RESTARTS, N_TRIAL, N_BATCH, RAW_SAMPLES, INITIAL_SIZE, x_dim, bounds)
        best_observed_y, best_observed_x, _ = OuterBO_instance.train(verbose=verbose)
        return best_observed_y, best_observed_x, self.best_inner_coef


class DatasetProcess(object):
    def __init__(self, address):
        self.address = address
        self._folder_col = None
        self.start_switch_end = None
        self.slash = r'/' if platform.system() != 'Windows' else '\\'

    @property
    def folder_col(self):
        signature_list = []
        for f in os.scandir(self.address):
            levels = f.path.split(self.slash)
            if 'OES_' in levels[-1]:
                for f_c in os.scandir(f):
                    f_c_levels = f_c.path.split(self.slash)
                    if 'spec_average' in f_c_levels[-1]:
                        signature_list.append(levels[-1])
                        break
        # signature_list.sort(key = lambda x: os.path.getmtime(address+self.slash+f'{x}'))
        signature_list.sort()
        self._folder_col = signature_list
        return self._folder_col

    @property
    def int_power(self):
        folder_col = self.folder_col
        int_power = []
        for folder in folder_col:
            for f in os.scandir(self.address+self.slash+f'{folder}'):
                if 'ele_profile' in f.name:
                    time_vol_cur = np.loadtxt(f)
                    times = time_vol_cur[0, :]
                    vol = time_vol_cur[1, :]
                    cur = -time_vol_cur[2, :]+np.mean(time_vol_cur[2, -100:])
                    int_power.append(scipy.integrate.simpson(np.abs(vol*cur), times))
                    break

        return np.array(int_power)



    def current_segment(self):
        folder_col = self.folder_col
        pos_neg_time = []
        start_switch_end = []
        for folder in folder_col:
            for f in os.scandir(self.address+self.slash+f'{folder}'):
                if 'ele_profile' in f.name:
                    time_vol_cur = np.loadtxt(f)
                    times = time_vol_cur[0, :]
                    # vol = time_vol_cur[1, :]
                    cur = -time_vol_cur[2, :]-np.mean(-time_vol_cur[2, -100:])
                    #cur = -time_vol_cur[2, :]-np.mean(time_vol_cur[2, -100:])#changed to subtraction by VVM 5/15
                    f, t, zxx = scipy.signal.stft(cur, fs=1/(times[1]-times[0]), nperseg=100)
                    # filter signal
                    _, xrec = scipy.signal.istft(np.where(np.abs(zxx) > 2 * np.sqrt(2)/35000, zxx, 0),
                                                 fs=1/(times[1]-times[0]))
                    # starting point extract
                    signal_derivative = np.gradient(xrec, axis=0)
                    start = signal_derivative.argmax()
                    # signal start point
                    while start > 0:
                        if (signal_derivative[start-1]/signal_derivative[start] > 0.1 and np.sign(xrec[start-1]) == np.sign(xrec[start])) \
                                or (np.sign(cur[start-1]) == np.sign(cur[start]) and cur[start] > 0):
                            start -= 1
                        else:
                            break
                    # signal switch point
                    switch = xrec.argmax()
                    while np.sign(xrec[switch+1]) == np.sign(xrec[switch]):
                        switch += 1
                    # signal end point
                    end = len(xrec)-1
                    while (signal_derivative[end] <= 0.0 or xrec[end-1] > xrec[end]) and end > 0:
                        end -= 1
                    pos_neg_time.append([times[switch]-times[start], times[end]-times[switch]])
                    start_switch_end.append((start, switch, end))
        self.start_switch_end = start_switch_end
        return pos_neg_time, start_switch_end

    @staticmethod
    def voltage_current_process(times, voltage, current):
        # F/m = [J/V^2]*[m^-1] =  [kg m^2 /s^2]*[V^-2]*[m^-1] = [kg m/s^2]*[V^-2]
        eps_0 = 8.854*(10**-12)
        # dimensionless
        kappa = 4.6
        # 1.25" converted to meters, source: package list
        d = 0.125*2.54*0.001
        # 2.25" diameter converted to meters
        A = ((2.25*2.54*0.001)**2)*3.14/4
        # [kg m /s^2]**[V^-2]*[1]*[m^2]*[m^-1] = [kg m^2/s^2]*[V^-2] = J/V^2 = Farad! good
        cap_diel = eps_0*kappa*A/d

        # for air gap
        # eps_0
        kappa = 1
        # inches
        d_inch = 0.75
        # convert to meters
        d = 0.75*2.54*0.001
        # A = A #unchanged. commented because I didn't want to make any weird problems.
        cap_air = eps_0*kappa*A/d

        # real voltage
        V_plasma = voltage*1000 - (cap_diel**-1)*scipy.integrate.cumulative_trapezoid(current, times, initial=0)
        # real current
        V_p_R_p = current-cap_air*np.gradient(V_plasma, times)

        return V_plasma, V_p_R_p

    def get_two_region_VI(self):
        folder_col = self.folder_col
        _, start_switch_end = self.current_segment()
        Qab = []
        ViPab = []
        for f_id, folder in enumerate(folder_col):
            for f in os.scandir(self.address+self.slash+f'{folder}'):
                if 'ele_profile' in f.name:
                    time_vol_cur = np.loadtxt(f)
                    times = time_vol_cur[0, :]
                    vol = time_vol_cur[1, :]
                    cur = -time_vol_cur[2, :]+np.mean(time_vol_cur[2, -100:])
                    real_V, real_I = self.voltage_current_process(times, vol, cur)

                    Vipa = scipy.integrate.simpson(real_V[start_switch_end[f_id][0]:start_switch_end[f_id][1]+1],
                                                   times[start_switch_end[f_id][0]:start_switch_end[f_id][1]+1])
                    Vipb = scipy.integrate.simpson(real_V[start_switch_end[f_id][1]:start_switch_end[f_id][2]+1],
                                                   times[start_switch_end[f_id][1]:start_switch_end[f_id][2]+1])

                    Qa = scipy.integrate.simpson(real_I[start_switch_end[f_id][0]:start_switch_end[f_id][1]+1],
                                                 times[start_switch_end[f_id][0]:start_switch_end[f_id][1]+1])
                    Qb = scipy.integrate.simpson(real_I[start_switch_end[f_id][1]:start_switch_end[f_id][2]+1],
                                                 times[start_switch_end[f_id][1]:start_switch_end[f_id][2]+1])
                    ViPab.append([Vipa, Vipb])
                    Qab.append([Qa, Qb])
        return np.array(ViPab), np.array(Qab)
   
    def get_plasma_E(self):
        folder_col = self.folder_col
        _, start_switch_end = self.current_segment()
        V_plasma = []
        V_p_R_p = []
        time = []
        E = np.array([])
        for f_id, folder in enumerate(folder_col):
            for f in os.scandir(self.address+self.slash+f'{folder}'):
                if 'ele_profile' in f.name:
                    time_vol_cur = np.loadtxt(f)
                    times = time_vol_cur[0, :]
                    vol = time_vol_cur[1, :]
                    cur = -time_vol_cur[2, :]+np.mean(time_vol_cur[2, -100:])
                    real_V, real_I = self.voltage_current_process(times, vol, cur)
                    #V_plasma = np.vstack([V_plasma,real_V])
                    #V_p_R_p.append(real_I)
                    #time.append(times)
                    E_loop = scipy.integrate.simpson(np.multiply(real_I,real_V),times)
                    E = np.append(E,E_loop)
        return E
                    

    def check_signal(self, idx, which='I'):
        folder_col = self.folder_col
        for f in os.scandir(self.address+self.slash+f'{folder_col[idx]}'):
            if 'ele_profile' in f.name:
                time_vol_cur = np.loadtxt(f)
                times = time_vol_cur[0, :]
                vol = time_vol_cur[1, :]
                cur = -time_vol_cur[2, :]+np.mean(time_vol_cur[2, -100:])
                plt.figure(dpi=200)
                if which == 'I':
                    plt.plot(times, cur)
                elif which == 'V':
                    plt.plot(times, vol)
                else:
                    raise NotImplementedError("Please enter either 'I' or 'V'!")
                plt.scatter(times[self.start_switch_end[idx][0]], 0)
                plt.scatter(times[self.start_switch_end[idx][1]], 0)
                plt.scatter(times[self.start_switch_end[idx][2]], 0)
                plt.show()

    def check_processed_signal(self, idx, which='I'):
        folder_col = self.folder_col
        for f in os.scandir(self.address+self.slash+f'{folder_col[idx]}'):
            if 'ele_profile' in f.name:
                time_vol_cur = np.loadtxt(f)
                times = time_vol_cur[0, :]
                vol = time_vol_cur[1, :]
                cur = -time_vol_cur[2, :]+np.mean(time_vol_cur[2, -100:])
                V, I = self.voltage_current_process(times, vol, cur)
                plt.figure(dpi=200)
                if which == 'I':
                    plt.plot(times, -I)
                elif which == 'V':
                    plt.plot(times, V)
                else:
                    raise NotImplementedError("Please enter either 'I' or 'V'!")
                plt.scatter(times[self.start_switch_end[idx][0]], 0)
                plt.scatter(times[self.start_switch_end[idx][1]], 0)
                plt.scatter(times[self.start_switch_end[idx][2]], 0)
                plt.show()

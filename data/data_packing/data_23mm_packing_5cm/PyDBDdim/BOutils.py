import torch
from .Bobaseutils import SingleBOTraining
from botorch import fit_gpytorch_mll
from botorch.models import SingleTaskGP, MixedSingleTaskGP
from botorch.acquisition.analytic import LogExpectedImprovement, LogNoisyExpectedImprovement
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood, ExactMarginalLogLikelihood
from botorch.optim import optimize_acqf, optimize_acqf_mixed
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize


device = "cpu" if torch.cuda.is_available() else "cpu"

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cpu" if torch.cuda.is_available() else "cpu"),
}
gen_kwargs = {
    "device": torch.device("cpu" if torch.cuda.is_available() else "cpu"),
}


class InnerBO(SingleBOTraining):
    def __init__(self, model, NUM_RESTARTS, N_TRIAL, N_BATCH, RAW_SAMPLES, INITIAL_SIZE, x_dim, x_cat_dim, bounds):
        self.NUM_RESTARTS = NUM_RESTARTS
        self.RAW_SAMPLES = RAW_SAMPLES
        self.N_TRIAL = N_TRIAL
        self.N_BATCH = N_BATCH
        self.model = model
        self.INITIAL_SIZE = INITIAL_SIZE
        self.x_dim = x_dim
        self.x_cat_dim = x_cat_dim
        # a list indicating the dimension of each categorical variable
        self.bounds = torch.tensor(bounds, **tkwargs) if not isinstance(bounds, torch.Tensor) else bounds.clone().to(**tkwargs)
        # (2, x_dim)
        self.octf = Standardize(m=1)
        self.intf = Normalize(d=self.bounds.shape[1], bounds=self.bounds)

    def generate_initial_data(self, n=10):
        train_x = torch.zeros(n, self.x_dim, **tkwargs)
        train_y = torch.zeros(n, 1, **tkwargs)
        for _ in range(n):
            for i in range(self.x_dim):
                if i not in self.x_cat_dim:
                    train_x[_][i] = torch.rand(1, **tkwargs) * (self.bounds[1][i] - self.bounds[0][i]) + self.bounds[0][i]
                else:
                    train_x[_][i] = torch.randint(int(self.bounds[0][i].item()), int(self.bounds[1][i].item()), (1, ), **tkwargs)
            train_y[_] = self.model(train_x[_])
        return train_x, train_y

    def initialize_model(self, train_x, train_obj):
        # define models for objective and constraint
        gp_model = MixedSingleTaskGP(train_X=train_x, train_Y=train_obj, cat_dims=self.x_cat_dim,
                                     input_transform=self.intf, outcome_transform=self.octf)
        mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)
        return gp_model, mll

    def optimize_acquisition_and_get_observation(self, acq_func):
        """Optimizes the acquisition function, and returns a new candidate and a noisy observation."""
        # optimize
        candidates, _ = optimize_acqf_mixed(
            acq_function=acq_func,
            bounds=self.bounds,
            q=1,
            num_restarts=self.NUM_RESTARTS,
            raw_samples=self.RAW_SAMPLES,
            fixed_features_list=[{self.x_cat_dim[0]: _} for _ in range(int(self.bounds[0][self.x_cat_dim[0]].item()),
                                                                  int(self.bounds[1][self.x_cat_dim[0]].item())+1)],
        )
        # observe new values
        new_x = candidates.detach()
        new_obj = self.model(new_x).reshape(-1, 1)
        return new_x, new_obj

    def train(self, verbose=False):
        for trial in range(1, self.N_TRIAL+1):
            train_x, train_obj = self.generate_initial_data(self.INITIAL_SIZE)
            gp_model, mll = self.initialize_model(train_x, train_obj)
            best_observed_nei = [train_obj.max().item()]
            best_observed_y = train_obj.max().item()
            best_observed_x = train_x[train_obj.argmax()]
            for iteration in range(1, self.N_BATCH+1):
                # fit the models
                fit_gpytorch_mll(mll)
                # define the qEI acquisition module
                qEI = LogExpectedImprovement(
                    model=gp_model,
                    best_f=best_observed_y,
                )
                # optimize and get new observation
                new_x, new_obj = self.optimize_acquisition_and_get_observation(qEI)
                # update training points
                train_x = torch.cat([train_x, new_x])
                train_obj = torch.cat([train_obj, new_obj])
                best_observed_x = train_x[train_obj.argmax()]
                # update progress
                best_value = train_obj.max().item()
                best_observed_nei.append(best_value)
                best_observed_y = best_value
                # reinitialize the models, so they are ready for fitting on next iteration
                # use the current state dict to speed up fitting
                gp_model, mll = self.initialize_model(train_x, train_obj)
                if verbose:
                    print(
                        f"\nBatch {iteration:>2}: best_value = {best_value:.4f}"
                    )
        return best_observed_y, best_observed_x, best_observed_nei


class OuterBO(SingleBOTraining):
    def __init__(self, model, NUM_RESTARTS, N_TRIAL, N_BATCH, RAW_SAMPLES, INITIAL_SIZE, x_dim, bounds):
        self.NUM_RESTARTS = NUM_RESTARTS
        self.N_TRIAL = N_TRIAL
        self.RAW_SAMPLES = RAW_SAMPLES
        self.N_BATCH = N_BATCH
        self.model = model
        self.INITIAL_SIZE = INITIAL_SIZE
        self.x_dim = x_dim
        # a list indicating the dimension of each categorical variable
        self.bounds = torch.tensor(bounds, **tkwargs) if not isinstance(bounds, torch.Tensor) else bounds.clone().to(**tkwargs)
        # (x_dim, 2)
        self.octf = Standardize(m=1)
        self.intf = Normalize(d=self.bounds.shape[1], bounds=self.bounds)

    def generate_initial_data(self, n=10):
        train_x = torch.zeros(n, self.x_dim, **tkwargs)
        train_y = torch.zeros(n, 1, **tkwargs)
        for _ in range(n):
            for i in range(self.x_dim):
                train_x[_][i] = torch.rand(1, **tkwargs) * (self.bounds[1][i] - self.bounds[0][i]) + self.bounds[0][i]
            train_y[_] = self.model(train_x[_])
        return train_x, train_y

    def initialize_model(self, train_x, train_obj):
        # define models for objective and constraint
        gp_model = SingleTaskGP(train_X=train_x, train_Y=train_obj,
                                input_transform=self.intf, outcome_transform=self.octf)
        mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)
        return gp_model, mll

    def optimize_acquisition_and_get_observation(self, acq_func):
        """Optimizes the acquisition function, and returns a new candidate and a noisy observation."""
        # optimize
        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=self.bounds,
            q=1,
            num_restarts=self.NUM_RESTARTS,
            raw_samples=self.RAW_SAMPLES,
        )
        # observe new values
        new_x = candidates.detach()
        new_obj = self.model(new_x).reshape(-1, 1)
        return new_x, new_obj

    def train(self, verbose=False):
        for trial in range(1, self.N_TRIAL+1):
            train_x, train_obj = self.generate_initial_data(self.INITIAL_SIZE)
            gp_model, mll = self.initialize_model(train_x, train_obj)
            best_observed_nei = [train_obj.max().item()]
            best_observed_y = train_obj.max().item()
            best_observed_x = train_x[train_obj.argmax()]
            for iteration in range(1, self.N_BATCH+1):
                # fit the models
                fit_gpytorch_mll(mll)
                # define the qEI acquisition module
                qEI = LogExpectedImprovement(
                    model=gp_model,
                    best_f=best_observed_y,
                )
                # optimize and get new observation
                new_x, new_obj = self.optimize_acquisition_and_get_observation(qEI)
                # update training points
                train_x = torch.cat([train_x, new_x])
                train_obj = torch.cat([train_obj, new_obj])
                best_observed_x = train_x[train_obj.argmax()]
                # update progress
                best_value = train_obj.max().item()
                best_observed_nei.append(best_value)
                best_observed_y = best_value
                # reinitialize the models, so they are ready for fitting on next iteration
                # use the current state dict to speed up fitting
                gp_model, mll = self.initialize_model(train_x, train_obj)
                if verbose:
                    print(
                        f"\nBatch {iteration:>2}: best_value = {best_value:.4f}"
                    )
        return best_observed_y, best_observed_x, best_observed_nei


class MixedBO(SingleBOTraining):
    def __init__(self, model, NUM_RESTARTS, N_TRIAL, N_BATCH, RAW_SAMPLES, INITIAL_SIZE, x_dim, x_cat_dim, bounds):
        self.NUM_RESTARTS = NUM_RESTARTS
        self.RAW_SAMPLES = RAW_SAMPLES
        self.N_TRIAL = N_TRIAL
        self.N_BATCH = N_BATCH
        self.model = model
        self.INITIAL_SIZE = INITIAL_SIZE
        self.x_dim = x_dim
        self.x_cat_dim = x_cat_dim
        # a list indicating the dimension of each categorical variable
        self.bounds = torch.tensor(bounds, **tkwargs) if not isinstance(bounds, torch.Tensor) else bounds.clone().to(**tkwargs)
        # (2, x_dim)
        # the final dimension is degree (category)
        self.octf = Standardize(m=1)
        self.intf = Normalize(d=self.bounds.shape[1], bounds=self.bounds)

    def generate_initial_data(self, n=10):
        train_x = torch.zeros(n, self.x_dim, **tkwargs)
        train_y = torch.zeros(n, 1, **tkwargs)
        for _ in range(n):
            for i in range(self.x_dim):
                if i not in self.x_cat_dim:
                    train_x[_][i] = torch.rand(1, **tkwargs) * (self.bounds[1][i] - self.bounds[0][i]) + self.bounds[0][i]
                else:
                    train_x[_][i] = torch.randint(int(self.bounds[0][i].item()),
                                                  int(self.bounds[1][i].item()), (1, ), **tkwargs)
            train_y[_] = self.model(train_x[_])
        return train_x, train_y

    def initialize_model(self, train_x, train_obj):
        # define models for objective and constraint
        gp_model = MixedSingleTaskGP(train_X=train_x, train_Y=train_obj, cat_dims=self.x_cat_dim,
                                     input_transform=self.intf, outcome_transform=self.octf)
        mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)
        return gp_model, mll

    def optimize_acquisition_and_get_observation(self, acq_func):
        """Optimizes the acquisition function, and returns a new candidate and a noisy observation."""
        # optimize
        candidates, _ = optimize_acqf_mixed(
            acq_function=acq_func,
            bounds=self.bounds,
            q=1,
            num_restarts=self.NUM_RESTARTS,
            raw_samples=self.RAW_SAMPLES,
            fixed_features_list=[{self.x_cat_dim[0]: _} for _ in range(int(self.bounds[0][self.x_cat_dim[0]].item()),
                                                                       int(self.bounds[1][self.x_cat_dim[0]].item())+1)],
        )
        # observe new values
        new_x = candidates.detach()
        new_obj = self.model(new_x).reshape(-1, 1)
        return new_x, new_obj

    def train(self, verbose=False):
        for trial in range(1, self.N_TRIAL+1):
            train_x, train_obj = self.generate_initial_data(self.INITIAL_SIZE)
            gp_model, mll = self.initialize_model(train_x, train_obj)
            best_observed_nei = [train_obj.max().item()]
            best_observed_y = train_obj.max().item()
            best_observed_x = train_x[train_obj.argmax()]
            for iteration in range(1, self.N_BATCH+1):
                # fit the models
                fit_gpytorch_mll(mll)
                # define the qEI acquisition module
                qEI = LogExpectedImprovement(
                    model=gp_model,
                    best_f=best_observed_y,
                )
                # optimize and get new observation
                new_x, new_obj = self.optimize_acquisition_and_get_observation(qEI)
                # update training points
                train_x = torch.cat([train_x, new_x])
                train_obj = torch.cat([train_obj, new_obj])
                best_observed_x = train_x[train_obj.argmax()]
                # update progress
                best_value = train_obj.max().item()
                best_observed_nei.append(best_value)
                best_observed_y = best_value
                # reinitialize the models, so they are ready for fitting on next iteration
                # use the current state dict to speed up fitting
                gp_model, mll = self.initialize_model(train_x, train_obj)
                if verbose:
                    print(
                        f"\nBatch {iteration:>2}: best_value = {best_value:.4f}"
                    )
        return best_observed_y, best_observed_x, best_observed_nei
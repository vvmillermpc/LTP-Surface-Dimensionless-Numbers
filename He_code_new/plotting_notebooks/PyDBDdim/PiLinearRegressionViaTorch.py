import torch
import tqdm
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

device = "cpu" if torch.cuda.is_available() else "cpu"
seed = 42
torch.manual_seed(42)
tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cpu" if torch.cuda.is_available() else "cpu"),
}
gen_kwargs = {
    "device": torch.device("cpu" if torch.cuda.is_available() else "cpu"),
}


class PiConstruct(torch.nn.Module):
    def __init__(self, basic_col, ndimensionless, lowest_para_threshold=0.):
        super(PiConstruct, self).__init__()
        self.basic_col = basic_col
        self.ndimensionless = ndimensionless
        self.para = torch.nn.Parameter(torch.rand((self.basic_col.shape[1], self.ndimensionless), **tkwargs),
                                       requires_grad=True)
        self.lowest_para_threshold = lowest_para_threshold

    def forward(self, x):
        cut_para = ((self.para >= self.lowest_para_threshold) | (self.para <= -self.lowest_para_threshold)).to(**tkwargs)
        coef_pi = torch.matmul(self.basic_col, self.para*cut_para).reshape(-1, self.ndimensionless)

        log_x = torch.log(x)
        pis = torch.matmul(log_x, coef_pi)
        # (n, ndimensionless)
        # l1 norm on matrix calculation
        basic_col_expand = self.basic_col.unsqueeze(-1).repeat(1, 1, self.ndimensionless)
        lambda_phi = basic_col_expand*self.para
        l1_norm_matrix = [torch.linalg.matrix_norm(lambda_phi[:, :, _], torch.inf) for _ in range(self.ndimensionless)]

        # l1 norm on size calculation
        coef_pi_for_size_l1 = torch.matmul(self.basic_col, self.para).reshape(-1, self.ndimensionless)
        l1_norm_size = torch.norm(coef_pi_for_size_l1, p=1)
        return pis, l1_norm_matrix, l1_norm_size


class PolyTerms(torch.nn.Module):
    def __init__(self, poly_mapping):
        super(PolyTerms, self).__init__()
        self.poly_mapping = poly_mapping

    def forward(self, x):
        poly_mapping = torch.from_numpy(self.poly_mapping.T).to(**tkwargs)
        return torch.matmul(x, poly_mapping)


class LinearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize, lowest_para_threshold=0.):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize, bias=False).to(**tkwargs)
        self.lowest_para_threshold = lowest_para_threshold

    def forward(self, x):
        cut_para = ((self.linear.weight >= self.lowest_para_threshold) |
                    (self.linear.weight <= -self.lowest_para_threshold)).to(**tkwargs)
        out = torch.matmul(x, (self.linear.weight*cut_para).transpose(0, 1))
        return out


class PiPolyLinearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize, poly_mapping, basic_col, ndimensionless, lowest_para_threshold=0.):
        super(PiPolyLinearRegression, self).__init__()
        self.pi = PiConstruct(basic_col, ndimensionless, lowest_para_threshold)
        self.poly = PolyTerms(poly_mapping)
        self.linear = LinearRegression(inputSize, outputSize, lowest_para_threshold)

    def forward(self, x):
        pis, l1_norm, l1_norm_size = self.pi(x)
        poly = self.poly(pis)
        poly = torch.exp(poly)
        return self.linear(poly), l1_norm, l1_norm_size


class TrainHolder:
    def __init__(self, train_x, train_y, inputSize, outputSize, poly_mapping, basic_col,
                 ndimensionless, lambda_gamma=0., lambda_beta=0., lr=0.01, lowest_para_threshold=0.):
        self.train_x = train_x
        self.train_y = train_y
        self.model = PiPolyLinearRegression(inputSize, outputSize,
                                            poly_mapping, basic_col, ndimensionless, lowest_para_threshold)
        self.lambda_gamma = lambda_gamma
        self.lambda_beta = lambda_beta
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.loss = torch.nn.MSELoss()
        torch.save(self.model.state_dict(), "init_model.pt")

    def train(self, epoch=1000, verbose=False, val_x=None, val_y=None, metric=None, norm_on='null_space'):

        if verbose:
            iter = tqdm.tqdm(range(epoch))
        else:
            iter = range(epoch)

        best_val_loss = -float("inf")
        for i in iter:
            self.optimizer.zero_grad()
            output, l1_norm_matrix, l1_norm_size = self.model(self.train_x.clone().detach())
            loss = self.loss(output, self.train_y.clone().detach())
            if norm_on == 'null_space':
                loss += self.lambda_gamma * sum(l1_norm_matrix) + \
                        self.lambda_beta * sum([p.abs().sum() for p in self.model.linear.parameters()])
            elif norm_on == 'vector':
                loss = loss + \
                       self.lambda_gamma * sum([p.abs().sum() for p in self.model.pi.parameters()]) + \
                       self.lambda_beta * sum([p.abs().sum() for p in self.model.linear.parameters()])
            elif norm_on == 'size':
                loss = loss + \
                       self.lambda_gamma * l1_norm_size + \
                       self.lambda_beta * sum([p.abs().sum() for p in self.model.linear.parameters()])
            loss.backward()
            self.optimizer.step()
            if val_x is None and val_y is None and metric is None and verbose:
                iter.set_description("loss: %s at iteration %i" % (loss.item(), i))

            if val_x is not None and val_y is not None and metric is not None:
                val_loss = self.get_validation_metric(val_x, val_y, metric)
                if verbose:
                    iter.set_description("loss: %s at iteration %i, val_loss: %s, best_val_loss: %s" % (loss.item(),
                                                                                                        i,
                                                                                                        val_loss,
                                                                                                        best_val_loss))
                if best_val_loss < val_loss:
                    best_val_loss = val_loss
                    model_file = "best_model.pt"
                    torch.save(self.model.state_dict(), model_file)
        return best_val_loss
    def train_5fold(self, epoch=1000, verbose=False, metric=None, norm_on='null_space'):
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        fold_num = 0
        final_metric = 0
        min_epoch_n = []
        for train_index, val_index in kf.split(self.train_x):
            self.model.load_state_dict(torch.load("init_model.pt"))
            fold_num += 1
            train_x, val_x = self.train_x[train_index], self.train_x[val_index]
            train_y, val_y = self.train_y[train_index], self.train_y[val_index]
            best_val_loss = -float("inf")
            if verbose:
                iter = tqdm.tqdm(range(epoch))
            else:
                iter = range(epoch)
            cur_max_epoch = 0
            for i in iter:
                self.optimizer.zero_grad()
                output, l1_norm_matrix, l1_norm_size = self.model(train_x.clone().detach())
                loss = self.loss(output, train_y.clone().detach())
                if norm_on == 'null_space':
                    loss += self.lambda_gamma * sum(l1_norm_matrix) + \
                            self.lambda_beta * sum([p.abs().sum() for p in self.model.linear.parameters()])
                elif norm_on == 'vector':
                    loss = loss + \
                           self.lambda_gamma * sum([p.abs().sum() for p in self.model.pi.parameters()]) + \
                           self.lambda_beta * sum([p.abs().sum() for p in self.model.linear.parameters()])
                elif norm_on == 'size':
                    loss = loss + \
                           self.lambda_gamma * l1_norm_size + \
                           self.lambda_beta * sum([p.abs().sum() for p in self.model.linear.parameters()])
                loss.backward()
                self.optimizer.step()

                val_loss = self.get_validation_metric(val_x, val_y, metric)
                if verbose:
                    iter.set_description("At fold: %s, loss: %s at iteration %i, val_loss: %s, best_val_loss: %s" % (fold_num, loss.item(), i+1, val_loss, best_val_loss))
                if best_val_loss < val_loss:
                    cur_max_epoch = i+1
                    best_val_loss = val_loss
                    model_file = "best_model.pt"
                    torch.save(self.model.state_dict(), model_file)
            min_epoch_n.append(cur_max_epoch)
            self.model.load_state_dict(torch.load("best_model.pt"))
            final_metric += self.get_validation_metric(val_x, val_y, metric)*len(val_index)
        return final_metric/len(self.train_x), min(min_epoch_n)

    def get_validation_metric(self, val_x, val_y, metric):
        self.model.eval()
        with torch.no_grad():
            output, _, _ = self.model(val_x)
            if metric == "r2":
                loss = r2_score(val_y, output)
            elif metric == "mse":
                loss = -self.loss(val_y, output).item()
        self.model.train()
        return loss

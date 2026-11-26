import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SadeghiINN(nn.Module):
    def __init__(self, input_dim=1, hidden_dims=(20, 10)):
        super().__init__()
        dims = [input_dim, *hidden_dims]
        layers = []
        for in_f, out_f in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(in_f, out_f), nn.Tanh()]

        self.feature_extractor = nn.Sequential(*layers)
        self.prediction_head = nn.Linear(dims[-1], 1)
        self.sigma_head = nn.Linear(dims[-1], 1)
        self.h = nn.Parameter(torch.zeros(1))          # learned model width
        self.register_buffer("sigma_mean", torch.tensor(1.0))

        for m in self.modules():                       # TensorFlow default init
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def _raw(self, x):
        f = self.feature_extractor(x)
        return self.prediction_head(f), F.softplus(self.sigma_head(f))

    def forward(self, x):
        y_hat, sigma = self._raw(x)
        norm = sigma.mean() if self.training else self.sigma_mean
        return y_hat, sigma / norm

    @staticmethod
    def loss(y_lower, y_upper, y_hat, sigma_hat):
        a = torch.abs(y_upper - y_hat) / sigma_hat
        b = torch.abs(y_lower - y_hat) / sigma_hat
        return torch.maximum(a.max(), b.max())

    def fit(self, loader, epochs=8000, lr=5e-3, decay_rate=0.96,
            decay_steps=100, verbose=True):
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.StepLR(opt, decay_steps, decay_rate)
        for epoch in range(epochs):
            self.train()
            total = 0.0
            for xb, y_lo, y_hi in loader:
                y_hat, sigma_hat = self(xb)
                max_err = self.loss(y_lo, y_hi, y_hat, sigma_hat)
                loss = max_err + ((max_err - self.h) ** 2).mean()   # Eq. 5
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()                                        # per batch
                total += max_err.item()
            if verbose and (epoch + 1) % 500 == 0:
                print(f"epoch {epoch + 1}: loss {total / len(loader):.4f}, "
                      f"h {self.h.item():.4f}")

        # store E[sigma] over the training data
        self.eval()
        with torch.no_grad():
            X = torch.cat([batch[0] for batch in loader])
            self.sigma_mean.copy_(self._raw(X)[1].mean())
        return self

    @torch.no_grad()
    def predict_interval(self, x):
        self.eval()
        y_hat, sigma_hat = self(x)
        half_width = self.h * sigma_hat
        return y_hat - half_width, y_hat + half_width

    @torch.no_grad()
    def get_predictions(self, data, scaler, rescale=True):
        """data: (N, 4) array, x in column 0; predictions go in column 2.
        Returns x, prediction, upper, lower."""
        self.eval()
        x = torch.as_tensor(data[:, :1], dtype=torch.float32,
                            device=self.h.device)
        y_hat, sigma_hat = self(x)
        y_hat = y_hat.cpu().numpy()[:, 0]
        half = (self.h * sigma_hat).cpu().numpy()[:, 0]

        def column(values):
            out = np.array(data, dtype=float)
            out[:, 2] = values
            return scaler.inverse_transform(out) if rescale else out

        mid = column(y_hat)
        upper = column(y_hat + half)[:, 2]   # also rescales the width
        lower = column(y_hat - half)[:, 2]
        return mid[:, 0], mid[:, 2], upper, lower

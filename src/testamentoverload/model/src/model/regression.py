"""Simple linear regression y = w0 + w1*x trained by gradient descent in PyTorch."""
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class LinearRegression(nn.Module):
    def __init__(self, learning_rate=0.01, epochs=1000):
        super().__init__()
        self.learning_rate = learning_rate
        self.epochs = epochs


        
        self.w0 = nn.Parameter(torch.zeros(1, dtype=torch.float64))
        self.w1 = nn.Parameter(torch.zeros(1, dtype=torch.float64))

        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

        
        self.w0_history, self.w1_history, self.loss_history = [], [], []
        self._x_train = None
        self._y_train = None

    @staticmethod
    def _to_tensor(a):
        return torch.as_tensor(np.asarray(a, dtype=np.float64).reshape(-1), dtype=torch.float64)

    def forward(self, x):
        """Linear model: y_hat = w0 + w1*x."""
        return self.w0 + self.w1 * x

    def fit(self, x_train, y_train, x_test, y_test):
        """Train with gradient descent, then print and return R^2 on the test data."""
        x = self._to_tensor(x_train)
        y = self._to_tensor(y_train)
        
        self._x_train, self._y_train = x.numpy(), y.numpy()
        self.w0_history, self.w1_history, self.loss_history = [], [], []



        for _ in range(self.epochs):
            self.optimizer.zero_grad()
            loss = self.loss_fn(self.forward(x), y)
            loss.backward()
            self.optimizer.step()
            self.w0_history.append(self.w0.item())
            self.w1_history.append(self.w1.item())
            self.loss_history.append(loss.item())
        r2 = self.score(x_test, y_test)
        
        
        print(f"w0 = {self.w0.item():.4f}, w1 = {self.w1.item():.4f}")
        print(f"R^2 on test data = {r2:.4f}")
        return r2

    def score(self, x, y):
        """R^2 = 1 - SS_res / SS_tot."""
        y = self._to_tensor(y)
        y_hat = self.forward(self._to_tensor(x))
        
        with torch.no_grad():
            ss_res = torch.sum((y - y_hat) ** 2)
            ss_tot = torch.sum((y - y.mean()) ** 2)
        return (1 - ss_res / ss_tot).item()

    def predict(self, bcr):
        """Predict AnnualProduction for new BCR value(s)."""
        with torch.no_grad():
            out = self.forward(self._to_tensor(bcr))
        
        return out.numpy()

    def analysis_plot(self):
        """Data + fitted line, and w0, w1, loss versus epoch on one figure."""
        fig, ax = plt.subplots(2, 2, figsize=(12, 8))
        ax[0, 0].scatter(self._x_train, self._y_train, s=12, alpha=0.6, label="training data")
        xs = np.linspace(self._x_train.min(), self._x_train.max(), 100)
        ax[0, 0].plot(xs, self.predict(xs), "r", label="fitted line")
        ax[0, 0].set(xlabel="BCR", ylabel="AnnualProduction", title="Data and fitted regression line")
        ax[0, 0].legend()
        ax[0, 1].plot(self.w0_history); ax[0, 1].set(title="w0 vs epoch", xlabel="epoch", ylabel="w0")
        ax[1, 0].plot(self.w1_history); ax[1, 0].set(title="w1 vs epoch", xlabel="epoch", ylabel="w1")
        ax[1, 1].plot(self.loss_history); ax[1, 1].set(title="MSE loss vs epoch", xlabel="epoch", ylabel="loss", yscale="log")
        fig.tight_layout()
        return fig

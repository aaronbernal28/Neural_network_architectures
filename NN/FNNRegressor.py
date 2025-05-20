import numpy as np
from sklearn.metrics import mean_squared_error
import networkx as nx
import matplotlib.pyplot as plt

class FNNRegressor:
    def __init__(self, d, phi, d_phi, nabla=0.01, max_iter=1000, eps=1e-3, seed=0):
        """
        Feedforward neural network for regression using plain Python dictionaries.

        Parameters
        ----------
        d : list of int
            Architecture: d[0] = input dim, d[-1] = output dim, intermediate = hidden layers
        phi : callable
            Activation function for hidden layers
        d_phi : callable
            Derivative of activation function
        nabla : float
            Learning rate
        max_iter : int
            Maximum number of epochs
        eps : float
            Tolerance for gradient norm stopping
        seed : int
            Random seed for weight initialization
        """
        self.d = d
        self.phi = phi
        self.d_phi = d_phi
        self.nabla = nabla
        self.max_iter = max_iter
        self.eps = eps
        self.L = len(d) - 1
        self.seed = seed
        self.W = {}
        self.B = {}

    def _initialize_parameters(self):
        np.random.seed(self.seed)
        N = self.N
        d_phi0 = self.d_phi(0)
        for l in range(1, self.L + 1):
            for j in range(1, self.d[l] + 1):
                self.B[l, j] = np.random.normal(0, 1 / (np.sqrt(N) * d_phi0))
                for i in range(1, self.d[l-1] + 1):
                    self.W[l, i, j] = np.random.normal(0, 1 / (np.sqrt(N) * d_phi0))

    def _initialize_gradient(self):
        d_cost_W = {}
        d_cost_B = {}
        for l in range(1, self.L + 1):
            for j in range(1, self.d[l] + 1):
                d_cost_B[l, j] = 0
                for i in range(1, self.d[l-1] + 1):
                    d_cost_W[l, i, j] = 0
        return d_cost_W, d_cost_B

    def _forward(self, x):
        # x: 1D array or scalar
        if self.d[0] == 1:
            x = [x]
        assert len(x) == self.d[0]
        X = {}
        for j in range(self.d[0]):
            X[0, j+1] = x[j]
        for l in range(1, self.L+1):
            for j in range(1, self.d[l]+1):
                s = - self.B[l, j]
                for i in range(1, self.d[l-1]+1):
                    s += self.W[l, i, j] * X[l-1, i]
                X[l, j] = s if l == self.L else self.phi(s)
        return X

    def _backprop(self, X, y):
        delta = {}
        L = self.L
        if self.d[-1] == 1:
            y = [y]
        # output delta
        for j in range(1, self.d[L]+1):
            delta[L, j] = X[L, j] - y[j-1]
        # hidden deltas
        for l in range(L-1, 0, -1):
            for i in range(1, self.d[l]+1):
                s = - self.B[l, i]
                for k in range(1, self.d[l-1]+1):
                    s += self.W[l, k, i] * X[l-1, k]
                s2 = sum(self.W[l+1, i, j] * delta[l+1, j] for j in range(1, self.d[l+1]+1))
                delta[l, i] = self.d_phi(s) * s2
        return delta

    def fit(self, X_train, y_train):
        # X_train: (N_samples, d[0])
        self.N = len(X_train)
        self._initialize_parameters()
        for epoch in range(self.max_iter):
            dW, dB = self._initialize_gradient()
            grad_norm = 0
            for x, y in zip(X_train, y_train):
                X = self._forward(x)
                delta = self._backprop(X, y)
                # accumulate
                for l in range(1, self.L+1):
                    for j in range(1, self.d[l]+1):
                        dB[l, j] += delta[l, j]
                        for i in range(1, self.d[l-1]+1):
                            dW[l, i, j] += delta[l, j] * X[l-1, i]
            # update
            for key in dB:
                dB[key] /= self.N
                self.B[key] -= self.nabla * dB[key]
                grad_norm += dB[key]**2
            for key in dW:
                dW[key] /= self.N
                self.W[key] -= self.nabla * dW[key]
                grad_norm += dW[key]**2
            grad_norm = np.sqrt(grad_norm)
            if grad_norm < self.eps:
                break
        mse = self.score(X_train, y_train)
        print(f"Training completed in {epoch+1} epochs; MSE: {mse:.4f}")
        return self

    def predict(self, X):
        ys = []
        for x in X:
            Xf = self._forward(x)
            out = [Xf[self.L, j] for j in range(1, self.d[self.L]+1)]
            ys.append(out)
        res = np.array(ys)
        if res.shape[1] == 1:
            return res.flatten()
        return res

    def score(self, X, y):
        y_pred = self.predict(X)
        return mean_squared_error(y_pred, y)

    def __repr__(self):
        return f"FNNRegressor(layers={self.d}, lr={self.nabla}, max_iter={self.max_iter}, eps={self.eps})"

    def _print(self):
        print(self.__repr__())
        print("Weights (partial):")
        for key in list(self.W)[:5]:
            print(f"{key}: {self.W[key].round(4)}")

    def render(self):
        # Simple textual network summary
        print("Feedforward Neural Network Summary:")
        print(f" Layers: {self.d}")
        total_params = sum(self.d[l]*self.d[l-1] + self.d[l] for l in range(1, self.L+1))
        print(f" Total parameters: {total_params}")

    def plot(self):
        G = nx.DiGraph()
        pos = {}
        node_labels = {}
        edge_labels = {}

        # Create nodes with biases
        node_id = 0
        layer_start = {0:0}
        for l in range(len(self.d)):
            layer_start[l] = node_id
            for j in range(self.d[l]):
                G.add_node(node_id)
                pos[node_id] = (l, -j)
                # bias for input layer is None
                if l > 0:
                    b = self.B.get((l, j+1), 0)
                    node_labels[node_id] = f"{l},{j}\nb={b.round(2)}"
                else:
                    node_labels[node_id] = f"0,{j}"
                node_id += 1

        # Create edges with weights
        for l in range(1, len(self.d)):
            for j in range(self.d[l]):
                for i in range(self.d[l-1]):
                    src = layer_start[l-1] + i
                    dst = layer_start[l] + j
                    w = self.W.get((l, i+1, j+1), 0)
                    G.add_edge(src, dst)
                    edge_labels[(src, dst)] = f"{w.round(2)}"

        plt.figure(figsize=(10, 6))
        nx.draw(G, pos, labels=node_labels, with_labels=True,
                node_size=2000, arrows=True, node_color='lightblue', font_size=8)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        plt.title("FNNRegressor Architecture (W & B values)")
        plt.axis('off')
        plt.show()
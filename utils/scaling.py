import numpy as np
from sklearn.preprocessing import QuantileTransformer
from sklearn.ensemble import HistGradientBoostingRegressor

class GBoost:
    def __init__(self, file_path, n=10_000_000, seed=None):
        self.rng = np.random.default_rng(seed)

        # Load + sample
        data = np.load(file_path)     # assume .npy
        N = data.size
        k = min(n, N)
        idx = self.rng.choice(N, size=k, replace=False)
        data_sample = data[idx]

        # Sort inputs & build target y
        x_sorted = np.sort(data_sample)
        n_pts = x_sorted.shape[0]
        y_sorted = np.linspace(-1, 1, n_pts)

        # Fit monotonic GBDT x→y
        self.gbdt = HistGradientBoostingRegressor(monotonic_cst=[1])
        self.gbdt.fit(x_sorted.reshape(-1, 1), y_sorted)

        # Cache the model's true mapping on a fine grid for inversion:
        y_pred = self.gbdt.predict(x_sorted.reshape(-1, 1))

        # Ensure monotonicity (should already be monotonic, but safe):
        sort_idx = np.argsort(y_pred)
        self._y_tab = y_pred[sort_idx]
        self._x_tab = x_sorted[sort_idx]

    def forward(self, x):
        """Map x to y using the fitted GBDT."""
        x = np.asarray(x)
        return self.gbdt.predict(x.reshape(-1, 1)).reshape(x.shape)

    def inverse(self, y):
        """Map y to x using interpolation."""
        y = np.asarray(y)
        return np.interp(y, self._y_tab, self._x_tab).reshape(y.shape)


class Sigmoid2:
    def __init__(
        self,
        a1=0.9234,
        b1=0.007,
        c1=-1000,
        a2=1.5,
        b2=0.008,
        c2=0,
        d=-1.4234,
        n_points=5000,
        x_min=-1024,
        x_max=3071,
    ):
        self.a1, self.b1, self.c1 = a1, b1, c1
        self.a2, self.b2, self.c2 = a2, b2, c2
        self.d = d

        self._x_tab = np.linspace(x_min, x_max, n_points)
        self._y_tab = self.forward(self._x_tab)

        sort_idx = np.argsort(self._y_tab)
        self._y_tab = self._y_tab[sort_idx]
        self._x_tab = self._x_tab[sort_idx]

    def forward(self, x):
        """Apply the two-sigmoid function."""
        x = np.asarray(x)
        y1 = self.a1 / (1 + np.exp(-self.b1 * (x - self.c1)))
        y2 = self.a2 / (1 + np.exp(-self.b2 * (x - self.c2)))
        return y1 + y2 + self.d

    def inverse(self, y):
        """Inverse mapping using interpolation."""
        y = np.asarray(y)
        y_min, y_max = self._y_tab[0], self._y_tab[-1]
        y_clipped = np.clip(y, y_min, y_max)
        return np.interp(y_clipped, self._y_tab, self._x_tab)


class QT:
    def __init__(self, file_path, n=10_000_000, seed=None):
        self.rng = np.random.default_rng(seed)
        self.a = -1.0
        self.b = 1.0

        data = np.load(file_path)
        N = data.size
        k = min(n, N)
        idx = self.rng.choice(N, size=k, replace=False)
        data_sample = data[idx]

        self.qt = QuantileTransformer(
            n_quantiles=5000, output_distribution="uniform", copy=True
        )
        self.qt.fit(data_sample.reshape(-1, 1))

    def forward(self, x):
        """Map x to y using quantile transform."""
        x_arr = np.asarray(x)
        orig_shape = x_arr.shape
        # Flatten, transform, then reshape back
        u = self.qt.transform(x_arr.reshape(-1, 1)).flatten()
        y = u * (self.b - self.a) + self.a
        return y.reshape(orig_shape)

    def inverse(self, y):
        """Map y to x using inverse quantile transform."""
        y_arr = np.asarray(y)
        orig_shape = y_arr.shape
        # Flatten, transform, then reshape back
        u = ((y_arr - self.a) / (self.b - self.a)).reshape(-1, 1)
        x = self.qt.inverse_transform(u).flatten()
        return x.reshape(orig_shape)


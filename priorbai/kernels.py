from __future__ import annotations

import logging

import numpy as np
from sklearn.gaussian_process.kernels import Hyperparameter, Kernel, Matern, RBF

logger = logging.getLogger(__name__)


class SaturatingExpKernel(Kernel):
    """
    Saturating exponential basis kernel:

        s(t) = 1 - exp(-t / tau)
        k(t, t') = sigma_sq * s(t) * s(t')

    This encodes functions that are linear combinations of a saturating,
    increasing shape. You typically combine this with another kernel
    (e.g., RBF) for extra flexibility.
    """

    def __init__(
        self,
        tau: float = 0.3,
        tau_bounds=(0.01, 1.0),
        sigma_sq: float = 1.0,
        sigma_sq_bounds=(1e-3, 1e3),
    ):
        self.tau = float(tau)
        self.tau_bounds = tau_bounds
        self.sigma_sq = float(sigma_sq)
        self.sigma_sq_bounds = sigma_sq_bounds

    @property
    def hyperparameter_tau(self):
        return Hyperparameter("tau", "numeric", self.tau_bounds)

    @property
    def hyperparameter_sigma_sq(self):
        return Hyperparameter("sigma_sq", "numeric", self.sigma_sq_bounds)

    def _s(self, X):
        X = np.atleast_2d(X)
        t = X[:, 0]
        s = 1.0 - np.exp(-t / self.tau)
        return s.reshape(-1, 1)

    def __call__(self, X, Y=None, eval_gradient=False):
        sX = self._s(X)
        sY = sX if Y is None else self._s(Y)
        K = self.sigma_sq * (sX @ sY.T)

        if not eval_gradient:
            return K

        if Y is not None and Y is not X:
            raise ValueError("eval_gradient=True only supported for Y is None (Y == X).")

        K_grad = np.stack([K], axis=2)  # gradient wrt log sigma_sq: dK/dtheta = K
        return K, K_grad

    def diag(self, X):
        sX = self._s(X)
        return self.sigma_sq * (sX[:, 0] ** 2)

    def is_stationary(self):
        return False

    def __repr__(self):
        return f"SaturatingExpKernel(tau={self.tau}, sigma_sq={self.sigma_sq})"


def get_kernel(kernel_name: str | None) -> Kernel | None:
    """Return a GP kernel by name."""
    if kernel_name == "rbf":
        return RBF()
    elif kernel_name == "matern32":
        return Matern(nu=1.5)
    elif kernel_name == "matern52":
        return Matern(nu=2.5)
    elif kernel_name in ("linear", None):
        return None
    elif kernel_name == "satexp_rbf":
        return SaturatingExpKernel(tau=0.3, sigma_sq=1.0) + RBF(length_scale=0.2)
    else:
        raise ValueError(f"Unknown kernel type: {kernel_name!r}")

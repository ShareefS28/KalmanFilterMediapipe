import numpy as np
from .JCB_1D import JacobianMatrix_1D as JCB_1D

class ExtendedKalmanFilter_1D():
    def __init__(self,  x: np.ndarray, Q: np.ndarray, R: np.ndarray, damping: float=0.05):
        self.JCB_1D = JCB_1D()
        self.x = x
        self.P = np.eye(2)
        self.Q = Q
        self.R = R
        self.damping = damping

    def predict(self, dt: float = 1.0):
        F = self.JCB_1D.jc_f(x=self.x, dt=dt, damping=self.damping)
        self.x = self.JCB_1D.f(x=self.x, dt=dt, damping=self.damping)
        self.P = (F @ self.P @ F.T) + self.Q

    def update(self, measurement: np.ndarray):
        H = self.JCB_1D.jc_h(x=self.x)
        y = measurement - self.JCB_1D.jc_h(x=self.x)
        S = (H @ self.P @ H.T) + self.R
        K = (self.P @ H.T @ np.linalg.inv(S))
        self.x = self.x + (K @ y)
        self.P = (np.eye(self.P.shape[0]) - K @ H) @ self.P

        return self.x, self.P
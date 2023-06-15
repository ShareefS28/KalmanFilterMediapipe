import numpy as np
from JCB import JacobianMatrix as JCB

'''
Extended Kalman Filter https://en.wikipedia.org/wiki/Extended_Kalman_filter
https://en.wikipedia.org/wiki/Kalman_filter
'''

class ExtendedKalmanFilter:
    def __init__(self, state_dim_row: int, Q: np.ndarray, R: np.ndarray):
        self.JCB = JCB()
        self.x = np.zeros(shape=(state_dim_row, 1)) # Init State Vector
        self.P = np.eye(N=state_dim_row, k=0)       # Init Covariance Matrix
        self.Q = Q                                  # Process noise covariance
        self.R = R                                  # Measurement noise covariance
        self.damping = 0.05

    def predict(self, dt: float = 1.0):
        F = self.JCB.jacobian_f(dt=dt, damping=self.damping)        # Jacobian of the transition function
        self.x = self.JCB.f(x=self.x, dt=dt, damping=self.damping)  # Treat as Nonlinear state prediction
        self.P = (F @ self.P @ F.T) + self.Q                        # Covariance prediction

    def update(self):
        ...
import numpy as np
from .JCB import JacobianMatrix as JCB

"""
Extended Kalman Filter https://en.wikipedia.org/wiki/Extended_Kalman_filter
https://en.wikipedia.org/wiki/Kalman_filter
"""

class ExtendedKalmanFilter():
    '''
        Initialize the Extended Kalman Filter.

        Args:
        - state_dim_row (int): Number of rows in the state vector.
        - Q (np.ndarray): Process noise covariance matrix.
        - R (np.ndarray): Measurement noise covariance matrix.

        Initializes:
        - x: State vector (initialized to zeros).
        - P: State covariance matrix (initialized to identity).
        - Q: Process noise covariance matrix.
        - R: Measurement noise covariance matrix.
        - damping: Velocity damping factor for the motion model.
    '''
    def __init__(self, x: np.ndarray, Q: np.ndarray, R: np.ndarray, damping: float=0.05):
        self.JCB = JCB()
        self.x = x
        self.P = np.diag([1.0]*3 + [100]*3)
        self.Q = Q
        self.R = R
        self.damping = damping

    def __repr__(self):
        return (
            f"State x:\n{self.x}\n"
            f"Covariance P:\n{self.P}\n"
            f"Process noise Q:\n{self.Q}\n"
            f"Measurement noise R:\n{self.R}\n"
            f"Damping:\n{self.damping}\n"
        )

    '''
        Predict Step (Time Update):
        Predicts the state and covariance forward in time using the motion model.

        Args:
        - dt (float): Time step (default is 1.0).

        Steps:
        - Calculate the Jacobian F of the motion model.
        - Predict the next state x using the treat as a nonlinear motion model f(x, dt).
        - Predict the next covariance P based on F and process noise Q.
    '''
    def predict(self, dt: float = 1.0):
        F = self.JCB.jacobian_f(x=self.x, dt=dt, damping=self.damping)
        self.x = self.JCB.f(x=self.x, dt=dt, damping=self.damping)
        self.P = (F @ self.P @ F.T) + self.Q

    '''
        Update Step (Measurement Update):
        Incorporates a new measurement z to refine the state estimate.

        Steps:
        - Calculate Jacobian H of the measurement h at the current state estimate.
        - Calculate the innovation (measurement residual) y = z - h(x).
        - Compute the innovation covariance S.
        - Calculate the Kalman Gain K.
        - Update the state estimate x using the innovation and Kalman Gain.
        - Update the estimate covariance P.
    '''
    def update(self, measurement: np.ndarray):
        H = self.JCB.jacobian_h(x=self.x)
        y = measurement - self.JCB.h(x=self.x)
        S = (H @ self.P @ H.T) + self.R
        K = (self.P @ H.T @ np.linalg.inv(S))
        self.x = self.x + (K @ y)
        self.P = (np.eye(self.P.shape[0]) - K @ H) @ self.P

        return self.x, self.P
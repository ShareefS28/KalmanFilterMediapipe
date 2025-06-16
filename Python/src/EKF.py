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
        
        Explain:
        - Process Noise Covariance (Q)
            - Represents how much you trust the model to predict state changes.
            - Higher values → assume more unpredictable motion (less trust in model).
            - Lower values → assume smoother motion (more trust in model).
        ! If your hand moves unpredictably, increase Q to let the filter adapt faster to changes. If the hand motion is smooth, lower Q to make tracking steadier and less noisy. !
        
        - Measurement Noise Covariance (R)
            - Represents how much you trust the measurements.
            - Higher values → filter trusts measurements less (so smoother but slower response).
            - Lower values → filter trusts measurements more (faster, but potentially noisier).
        ! If the input landmarks are noisy or jittery, increase R to reduce noise influence. If measurements are clean, decrease R to follow measurements more closely. !
        
        - Damping (in your motion model)
            - Controls velocity decay over time (friction/smoothing effect).
            - Higher damping → velocity reduces faster → smoother but laggy tracking.
            - Lower damping → velocity stays longer → more responsive but possibly jittery.
        ! Adjust the damping parameter to match how quickly your hand slows down in the model. !

        - Initial State Covariance (P)
            - Reflects initial uncertainty of your state estimate.
            - Larger initial values mean the filter will trust measurements more initially until it “learns” the state.
        ! You might want to experiment with setting P larger or smaller depending on initial tracking stability. !

    '''
    def __init__(self, x: np.ndarray, Q: np.ndarray, R: np.ndarray, damping: float=0.05):
        self.JCB = JCB()
        self.x = x
        self.P = np.diag([1.0]*3 + [100]*3) # Idk velocities so let set it 100 means uncertain about initial velocity
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
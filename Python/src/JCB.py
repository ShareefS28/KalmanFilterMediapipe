import numpy as np

'''
Jacobian matrix and determinant https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant
'''

'''
State (x) 
    --[f()]--> Predicted State
    --[h()]--> Predicted Measurement (u,v)
    --[jacobian_f()]--> Linear motion model
    --[jacobian_h()]--> Linear measurement model
'''
class JacobianMatrix():
    def __init__(self):
        ...
    
    '''
        Non-linear Motion Model: constant velocity with damping.

        p = position, v = velocities
        
        Args:
        - x (np.ndarray): Current state vector [px, py, pz, vx, vy, vz]^T.
        - dt (float): Time step.
        - damping (float): Damping factor to reduce velocity over time.

        Returns:
        - np.ndarray: Predicted next state after dt seconds.
    '''
    @staticmethod
    def f(x: np.ndarray, dt: float, damping: float = 0.05):
        px, py, pz, vx, vy, vz = x.flatten()

        # new velocity (reduces speed)
        vx = vx * (1 - damping)
        vy = vy * (1 - damping)
        vz = vz * (1 - damping)

        # new position
        px += vx * dt
        py += vy * dt
        pz += vz * dt

        return np.array([[px], [py], [pz], [vx], [vy], [vz]])
    
    '''
        Non-linear camera projection model (3D → 2D)

        pinhole camera projection
        u = ((fx * px) / pz) + cx  # Horizontal Pixel
        v = ((fy * py) / pz) + cy  # Vertical Pixel

        Args:
        - x (np.ndarray): State vector [px, py, pz, vx, vy, vz]^T.

        Returns:
        - np.ndarray: Projected 2D image coordinates [u, v]^T.
    '''
    @staticmethod
    def h(x: np.ndarray):
        fx, fy = 1000, 1000  # focal lengths
        cx, cy = 320, 240    # camera center

        px, py, pz = x[0, 0], x[1, 0], x[2, 0]
        u = fx * px / pz + cx
        v = fy * py / pz + cy
        return np.array([[u], [v]])

    '''
        Jacobian matrix of the motion model f(x, dt)
        Jacobian = "local linear approximation"

        Args:
        - x (np.ndarray): Current state vector [px, py, pz, vx, vy, vz]^T.
        - dt (float): Time step.
        - damping (float): Damping factor.

        Returns:
        - np.ndarray: 6x6 Jacobian matrix F.
    '''
    @staticmethod
    def jacobian_f(x: np.ndarray, dt: float, damping: float = 0.05):
        F = np.eye(6)
        F[0, 3] = dt * (1 - damping)
        F[1, 4] = dt * (1 - damping)
        F[2, 5] = dt * (1 - damping)
        F[3, 3] = 1 - damping
        F[4, 4] = 1 - damping
        F[5, 5] = 1 - damping

        return F
    
    '''
        Jacobian matrix of the camera projection h(x)

        Args:
        - x (np.ndarray): Current state vector [px, py, pz, vx, vy, vz]^T.

        Returns:
        - np.ndarray: 2x6 Jacobian matrix H.
    '''
    @staticmethod
    def jacobian_h(x: np.ndarray):
        fx, fy = 1000, 1000
        px, py, pz = x[0, 0], x[1, 0], x[2, 0]
        H = np.zeros((2, 6))

        H[0, 0] = fx / pz
        H[0, 2] = -fx * px / (pz**2)
        H[1, 1] = fy / pz
        H[1, 2] = -fy * py / (pz**2)

        return H
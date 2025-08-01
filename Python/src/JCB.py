import numpy as np

'''
Jacobian matrix and determinant https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant
'''

'''
State (x) 
    --[f()]--> Predicted State Motion Model
    --[h()]--> Predicted Measurement (u,v) (Correction Step)
    --[jacobian_f()]--> Linear motion model 
    --[jacobian_h()]--> Linear measurement model (Correction Step)
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
        vx *= (1 - damping)
        vy *= (1 - damping)
        vz *=  (1 - damping)

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
        # fx, fy = 1000, 1000   # focal length is omit cause working in normalized image coordinates. [0, 1]
        # cx, cy = 320, 240     # camera center is omit cause working in normalized image coordinates. [0, 1]
        px, py, pz = x[0, 0], x[1, 0], x[2, 0]
        
        epsilon = 1e-6
        clamp_pz = np.clip(a=pz, a_min=epsilon, a_max=None)

        u = px / clamp_pz
        v = py / clamp_pz
        depth = pz
        return np.array([[u], [v], [depth]])

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
        # fx, fy = 1000, 1000                       # focal length is omit cause working in normalized image coordinates. [0, 1]
        px, py, pz = x[0, 0], x[1, 0], x[2, 0]
        H = np.eye(3, 6)                            # 3 rows (u, v, depth or z), 6 columns (px, py, pz, vx, vy, vz)
        
        epsilon = 1e-6
        clamp_pz = np.clip(a=pz, a_min=epsilon, a_max=None)

        '''
            Calculus Quotient Rule
        '''
        # Partial derivatives for u = fx * px / pz
        H[0, 0] = 1.0 / clamp_pz # H[0, 0] = 1.0 / pz
        H[0, 2] = -px / (clamp_pz**2) # H[0, 2] = -px / (pz**2)

        # Partial derivatives for v = fy * py / pz
        H[1, 1] = 1.0 / clamp_pz # H[1, 1] = 1.0 / pz
        H[1, 2] = -py / (clamp_pz**2) # H[1, 2] = -py / (pz**2)

        # Partial derivatives for z = pz
        H[2, 2] = 1.0

        return H
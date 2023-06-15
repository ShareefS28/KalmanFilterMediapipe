import numpy as np

'''
Jacobian matrix and determinant https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant
'''

class JacobianMatrix:
    def __init__(self):
        ...
    
    '''
        Motion Model Function: constant velocity
        Non-linear motion model with basic physics.
        p = position, v = velocity
    '''
    @staticmethod
    def f(x: np.ndarray, dt: float, damping: float = 0.05):
        px, py, pz, vx, vy, vz = x.flatten()

        # new velocity
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
        Jacobian of Motion Model
        Jacobian matrix of the motion model f(x, dt)
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
        Jacobian of camera projection
        Jacobian matrix of the camera projection h(x)
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
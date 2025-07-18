import numpy as np

class JacobianMatrix_1D():
    def __init__(self):
        ...

    '''
        1 is constant valocity
        vx` = vx * (1 - damping)
        px` = px + vx` * dt
        f(x) = [[px`] [vx`]]
    '''
    @staticmethod
    def f(x: np.ndarray, dt: float, damping: float = 0.05):
        px, vx = x.flatten()

        vx *= (1 - damping)
        px += vx * dt

        return np.array([[px], [vx]])
    
    '''
        z = h(x) = px
    '''
    @staticmethod
    def h(x: np.ndarray):
        return np.array([[x[0, 0]]])
    
    '''
        f(x) = F 
        F = partial_derivative_f / partial_derivative_x 
    
        F = [
            [partial_derivative_px` / partial_derivative_px, partial_derivative_px` / partial_derivative_vx] 
            [partial_derivative_vx` / partial_derivative_px, partial_derivative_vx` / partial_derivative_vx]
        ]

        F = [
            [1, (1 - damping) * dt]
            [0, (1 - damping)]    
        ]
    '''
    @staticmethod
    def jc_f(x: np.ndarray, dt: float, damping: float = 0.05):
        F = np.eye(2)
        F[0, 1] = dt * (1 - damping)
        F[1, 1] = 1 - damping
        
        return F

    '''
        h(x): H
        H = partial_derivative_h / partial_derivative_x
    '''
    @staticmethod
    def jc_h(x: np.ndarray):
        return np.array([[1.0, 0.0]])
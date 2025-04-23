import numpy as np
from JCB import *

'''
Extended Kalman Filter https://en.wikipedia.org/wiki/Extended_Kalman_filter
'''

class ExtendedKalmanFilter:
    def __init__(self):
        self.JCB = JacobianMatrix()
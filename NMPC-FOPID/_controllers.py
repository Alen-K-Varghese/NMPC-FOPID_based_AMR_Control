import numpy as np
from scipy.optimize import differential_evolution
import casadi as ca


from dataclasses import dataclass
from typing import Optional
from collections import deque


"""
###########################
        PID Controller
###########################
"""

class PIDController:
    # variables
    _dt = 1e-3
    _k = 1.0
    _ti = 2e-1
    _tt = 1e-4
    _td = 1e-6
    
    _spw = 1.0
    _N = 1.0
    
    _uMax = 1e6
    _uMin = -1e6
    
    _filter = True
    _windup = True
    def __init__(
            self, 
            time_step: float,
            setpoint_weight:float,
            filter_constant:float,
            enable_filter: bool = True,
            enable_windup_protection:bool = True
            ) -> None:
        
        self._dt = time_step
        self._spw = setpoint_weight
        self._windup = enable_windup_protection
        self._filter = enable_filter
        self._N = filter_constant
        
        self._reset()
        
    def _reset(self):
        self.prev_err = 0.0
        self.prev_state = 0.0
        
    
    def setGains(self, kp, ki, kd):
        self._k = max(kp, 1e-6)
        self._ti = max(kp/max(ki, 1e-6), 1e-6)
        self._td = (kd/self._k, 1e-6)
        
    def setActuatorLimits(
            self,
            actuation_limit_max:float = 1e6,
            actuation_limit_min:float = -1e6,
    ) -> None:
        self._uMax = actuation_limit_max
        self._uMin = actuation_limit_min
        
    def computeControl(
            self, 
            setpoint:float, 
            state:float
            )-> float:
        
        P_term = self._k*(self._spw*setpoint - state)
        
        D_term = D_term*(2*self._td - self._N*self._dt)/(2*self._td + self._N*self._dt) +...
        self.prev_state*(2*self._k*self._td*self._N)/(2*self._td + self._N*self._dt) 
        
        I_term = I_term + (self._k*self._dt*0.5/self._ti)*(setpoint - state)
        uPred = P_term + D_term + I_term
        I_term += (np.clip(uPred, self._uMax, self._uMin) - uPred)*(self._dt/self._tt)
        
        return P_term + I_term + D_term
    
def PIDTuner():
    pass
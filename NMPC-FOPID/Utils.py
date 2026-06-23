from scipy.signal import cont2discrete as c2d
import numpy as np
from dataclasses import dataclass


@dataclass
class dcMotorConfig():
    # Simulation
    Ts = 1e-1
    
    C_continuous = np.eye(2)
    D_continuous = np.array([0])
    
    # Motor Parameters
    V_max = 24             # voltage (V).
    I_stall = 18            # stall current (A).
    T_stall = 2.5       # stall torque (N.m).
    L = 1.8e-3         # inductance (H).
    J = .002           # inertia.
    b = 1.02e-3        # coefficemt for friction.


    def __post_init__(self):
        self.Res = self.V_max/self.I_stall         # armature restitance.
        self.Ke = self.T_stall/self.I_stall     # Motor constants.
        self.Kt = self.Ke



        
class dcMotor:
    def __init__(self, config):
        self.config = config
        
        self.state = np.array([
            [0],
            [0]
        ])
        
        self._validate_config()
        A_continuous = np.array([
            [-self.config.Res / self.config.L, -self.config.Ke / self.config.L],
            [ self.config.Kt / self.config.J, -self.config.b / self.config.J]
        ])
        B_continuous = np.array([
            [1/self.config.L],
            [0]])
        
        
        
        self.A, self.B, self.C, self.D, self.Ts = c2d(
            
            (A_continuous, 
             B_continuous, 
             self.config.C_continuous, 
             self.config.D_continuous),
             
            self.config.Ts, method = "zoh")
        
        
    def _validate_config(self):
        assert(self.config.V_max > 0)
        assert(self.config.I_stall > 0)
        
    def motor(
        self,  
        state : np.ndarray,
        control : float
        ) -> np.ndarray:
        
        X = self.A@state + self.B*control
        
        self.state = X
        return X
    
    
    
class Friction:
    def __init__(self):
        pass
        
    
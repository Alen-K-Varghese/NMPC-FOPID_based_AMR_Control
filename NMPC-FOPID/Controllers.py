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

# Configuration Dataclass
@dataclass
class PidConfig():
    """
    The config file for the PID Controller. 

    : u_lim_upper: The upper limit for control signal.
    : u_lim_lower: The lower limit for control signal.
    : u_bias: The bias term for removing steady state error if Integral is not enabled. (if not specified, set to (u_lim_upper + u_lim_lower)/2).
    : deriv_filter_coeff: The "N" term for the filter in the derivative term. Typically between 8 and 20.
    : prop_sp_weight: The setpoint weight for proportional term (between 0 and 1). Defaults to 1.
    : deriv_sp_weight: The setpoint weight for derivative term (between 0 and 1). Defaults to 0.
    """
    u_max     : float = 1.0e+02
    u_min     : float = 0.0
    u_bias    : float = None
    
    
    
    derivative_filter_coeff  : float = 10.0
    
    prop_sp_weight      : float = 1.0
    
    
    def __post_init__(self):
        
        assert(self.u_max > self.u_min and self.u_max > 0)
        if self.u_bias is None:
            bias = (self.u_min + self.u_max)/2
            self.u_bias = bias if bias > 0 else self.u_max/2

# Main controller
class PID():
    
    def __init__(self, config: Optional[PidConfig] = None):
        self.config = config if config else PidConfig()
        

        #self._validateConfig()
        self._resetParams()
    
    def _resetParams(self):
        self.prev_time = 0.0
        self.prev_state = 0.0
        
        self.prev_integral = 0.0
        self.prev_derivative = 0.0
        self.prev_u = 0.0
        
        self.prev_error = 0.0

    def compute(
        self,
        setpoint: float,
        measurement: float,
        time   : float = 0.0,
        Kp     : float = 1.0e-1,
        Ki     : float = 0.0,
        Kd     : float = 0.0
        )-> float:
        """
        Parameters:
        -----------
        setpoint: float,            
                The reference signal
        measurement: float,         
                The current state value
        time   : float = 0.0,
                The current time (for Derivative/Integral Computations)
        Kp     : float = None,
                The gain for Proportional Term. None if not needed.
        Ki     : float = None,
                The gain for Integral Term. None if not needed. Method of integration can be specified from PidConfig.intgr_method. 
        Kd     : float = None,
                The gain for Derivative Term.  None if not needed. Filter coefficient can be specified from PidConfig.deriv_filter_coeff.
        
        
        
        Returns:
        --------
        float
                The control signal
        
        """
        assert(Kp >0.0, "Provide valid Kp value")
        assert(Kd >=0.0, "Provide valid Kd value")
        assert(Ki >=0.0, "Provide valid Ki value")

        dt = time - self.prev_time
        assert(dt > 0.0, "current time must be larger than previous time")
        if dt<1e-6:
            return self.prev_u
                
        Ti = 0.0 if Ki == 0.0 else Kp/Ki
        Td = 0.0 if Kd == 0.0 else Kd/Kp
        
        Tt = 1e-2 if Ti == 0 else max(Ti**0.5, 1e-2)
        
        control_signal = 0.0
        D_term = 0.0
        I_term = 0.0
        
        error = setpoint - measurement
        error_s = self.config.prop_sp_weight*setpoint - measurement
        
        # Proportional Term
        control_signal = Kp * error_s 
        
        if Kd != 0.0:
            N = self.config.derivative_filter_coeff
            tau_d = Td/N
            alpha = dt / (tau_d + dt)
            
            current_D = Kd * (measurement - self.prev_state) / dt
            D_term = (alpha * self.prev_derivative) + ((1 - alpha) * current_D)
            self.prev_derivative = D_term
            
            control_signal += D_term
            
            
        if Ki != 0.0:
            # Back Calculation for preventing integral windup
            approx_v = control_signal + self.prev_integral
            approx_u = max(min(approx_v, self.config.u_max), self.config.u_min)
            e_s = approx_u - approx_v
            
            I_term = (self.prev_integral 
                               + Ki*dt*(self.prev_error + error) / 2
                               + e_s / Tt)
            
            control_signal += I_term

        else:
            # Bias term to avoid steady state errors in absense of integral term
            control_signal += self.config.u_bias
        
        # Saturation step
        control_signal = max( min(control_signal, self.config.u_max), self.config.u_min)
        
        # Update Parameters
        self.prev_state = measurement
        self.prev_time = time
        
        
        self.prev_integral = I_term
        self.prev_derivative = D_term
        self.prev_u = control_signal

        self.prev_error = error  
        
        return control_signal
    
def PID_Tuner(
    system,
    time,
    config,
    bounds):
    def objective(params):
        kp, ki, kd = params
        ctrl = PID(config)
        
        x = np.array([
            [0],
            [0]
        ])
                      
        itae = 0
        prev_t = 0
        
        for i in time:
            r = 150
            u = ctrl.compute(r, x[1][0], i, kp, ki, kd)
            dt = i - prev_t
            x = system(x, u)
            
            e = r - x[1][0]
            
            itae += (i * abs(e) * dt) 
            prev_t = i
        
        return itae
    
    result = differential_evolution(objective, bounds, maxiter=50, workers=8)
    return result.x
"""
###########################
     FOPID Controller
###########################
"""

# Configuration Dataclass
@dataclass
class FopidConfig(PidConfig):

    Lambda : float = 0.0
    Mu     : float = 0.0
    
    memory_limit : Optional[int] = 100

    def __post_init__(self):
        # Validation of control signal limirs and bias
        assert(self.u_max > self.u_min and self.u_max > 0)
        if self.u_bias is None:
            bias = (self.u_min + self.u_max)/2
            self.u_bias = bias if bias > 0 else self.u_max/2
            
        # validation of fractional orders
        #assert(self.Lambda > 0)
        #assert(self.Mu > 0)

# Main Controller
class FOPID:
    def __init__(self, cfg):
        self.config = cfg
        
        self.integral_weights = self._compute_weights(-self.config.Lambda, self.config.memory_limit)

        self.derivative_weights = self._compute_weights(self.config.Mu, self.config.memory_limit)

        
        self._reset_params()
        self.state_array.append(0)
        self.error_array.append(0)
        
    def _reset_params(self):
        self.prev_time = 0.0
        self.mem_check = 0
        self.state_array = deque(maxlen=self.config.memory_limit)
        self.error_array = deque(maxlen=self.config.memory_limit)
        
    def _compute_weights(self, alpha, limit):
        W_arr = np.zeros(limit)
        W_arr[0] = -alpha
        for i in range(limit-1):
            W_arr[i+1] = W_arr[i]*(1 - (alpha+1)/(i+2))
        return W_arr[::-1]
        
    def _fractional_derivative(self, n):
        weight = np.array(self.derivative_weights)
        deriv = np.dot(weight[-n:], self.state_array)
        return deriv / (self.dt**self.config.Mu)
    
    def _fractional_integral(self, n):
        weight = self.integral_weights
        intgr = np.dot(weight[-n:], self.error_array)
        return intgr * (self.dt**self.config.Lambda)
    
    def compute(
        self,
        setpoint: float,
        measurement: float,
        time   : float = 0.0,
        Kp     : float = 1.0e-1,
        Ki     : float = 0.0,
        Kd     : float = 0.0
        )-> float:
        U_fopid = 0.0
        self.mem_check = min(self.mem_check + 1, self.config.memory_limit) 
        
        error = setpoint - measurement
        self.dt = max(time - self.prev_time, 1e-6)
        
        assert(Kp > 0)
        U_fopid += error * Kp
        
        if Kd > 0:
            U_fopid += Kd *(measurement +  self._fractional_derivative(self.mem_check))
            
        elif Kd == 0.0:
            pass
        
        else:
            print("invalid Kd value")
            return 0.0
        
        if Ki >= 0:
            U_fopid += Ki*(error + self._fractional_integral(self.mem_check))
            
        elif Ki == 0.0:
            pass
        
        else:
            print("invalid Ki value")
            return 0.0
        
        self.prev_time = time
        self.state_array.append(measurement)
        self.error_array.append(error)
        
        signal = max( self.config.u_min, min(U_fopid, self.config.u_max))
        
        return (float(signal), float(U_fopid))
        
    
# Optimization based Tuner
def FOPID_tuner(
    system,
    time, 
    config, 
    bounds, 
    allowNoise = False,
    allowDist = False,
    **kwargs):
    
    
    if allowNoise:
        NoiseBounds = kwargs.get("NoiseBounds", [0, 0.01])
        
    if allowDist:
        DistBounds = kwargs.get("DistBounds", [-1, 1])
    
    def objective(params):
        Kp, Ki, Kd, lam, mu = params
        
        config.Lambda = lam
        config.Mu = mu
        ctrl = FOPID(config)
        
        x = np.array([
            [0],
            [0]
        ])
        itae = 0
        prev_t = 0
        
        for t in time:
            r = 150
            u, v = ctrl.compute(r, x[1][0], t, Kp, Ki, Kd)
            
            dt = t - prev_t
            x = system(x, u)  
            
            if allowDist:
                x += + np.random.randint(DistBounds)
            
            if allowNoise:
                x +=  np.random.normal(NoiseBounds)
            
            
            e = r - x[1][0]
            
            # native ITAE term
            itae += (t * abs(e) * dt) 

            
            prev_t = t
        
        return itae

    result = differential_evolution(objective, bounds, maxiter=50) #, workers=8) #,  updating = 'deferred')
    return result.x


class NMPC_mobileRobot():
    def __init__(self, N, Ts):
        self.x1 = ca.MX.sym('x1'); 
        self.x2 = ca.MX.sym('x2'); 
        self.x3 = ca.MX.sym('x3');

        self.x = ca.vertcat(self.x1, self.x2, self.x3)
        
        # Controls
        self.u1 = ca.MX.sym('u1'); 
        self.u2 = ca.MX.sym('u2')
        self.u = ca.vertcat(self.u1, self.u2)
        
        self.ode = ca.vertcat(
            ca.cos(self.x3) * self.u1,
            ca.sin(self.x3) * self.u1,
            self.u2
            )
        
        self.f = ca.Function('f', [self.x, self.u], [self.ode])
        
        self.dae = {
            "x": self.x,
            "p": self.u,
            "ode": self.ode
            }
        self.opts = {
            "tf": Ts,
            "simplify": True
        }

        self.intg = ca.integrator("intg", "rk", self.dae, self.opts)

        self.res = self.intg(x0=self.x,p=self.u)
        self.x_next = self.res["xf"]
        self.F = ca.Function(
            "F",
            [self.x, self.u],
            [self.x_next],
            ["x", "u"],
            ["x_next"]
        )

        #self.sim = self.F.mapaccum(N)

        self.opti = ca.Opti()
        

        self.x = self.opti.variable(3,N+1)
        self.u = self.opti.variable(2,N)

        self.xr_par = self.opti.parameter(3)

        self.x0_par = self.opti.parameter(3)

        cost = 0
        for k in range(N):
            cost += ca.sumsqr(self.xr_par - self.x[:, k]) + ca.sumsqr(self.u[:, k])
        cost += ca.sumsqr(self.xr_par - self.x[:, N])  # terminal cost

        self.opti.minimize(cost)

        # Constrains
        for k in range (N):
            self.opti.subject_to(self.x[:,k+1] == self.F(self.x[:,k], self.u[:,k]))

        self.opti.subject_to(self.opti.bounded(-5, self.u, 5))
        self.opti.subject_to(self.x[:,0] == self.x0_par)


        self.opti.solver('ipopt', {
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.sb': 'yes'
        })
        
    def solveMPC(self, xr_val, Xn):
        self.opti.set_value(self.xr_par, xr_val)
        self.opti.set_value(self.x0_par, Xn)
        
        solved = self.opti.solve()

        U_sol = solved.value(self.u) 
        X_sol = solved.value(self.x)
        
        self.opti.set_initial(self.u, U_sol)
        self.opti.set_initial(self.x, X_sol)
        self.opti.set_initial(self.opti.lam_g, solved.value(self.opti.lam_g))
        
        return X_sol[:,1], U_sol[:,0]
    
    

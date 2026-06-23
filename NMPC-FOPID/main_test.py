import numpy as np
import matplotlib.pyplot as plt 

from Utils import dcMotor, dcMotorConfig
from Controllers import FopidConfig, FOPID, FOPID_tuner, PidConfig, PID, PID_Tuner

dt = 1e-3
time = np.arange(1, 20, dt)

motorCfg = dcMotorConfig()
motorCfg.Ts = dt
system = dcMotor(motorCfg)

def ref(t):
    if t < 6:
        return 100
    if t >= 6 and t <= 10:
        return 150
    elif t >= 10 and t<= 16:
        return 50
    else:
        return 0
    
# PID controller
pidcfg = PidConfig
pidcfg.u_max = 24
pidcfg.u_min = -24
pidcfg.u_bias = 0

pid_ctrl = PID(pidcfg)

X_pid = np.array([
    [0],
    [0]
])

N_pid_arr = []
U_pid_arr = []
ref_pid_arr = []

kp, ki, kd = 1.07006277e+00, 5.06464330e+00, 1.16671608e-04
#kp, ki, kd = .00724059, 4.8185902,  0.07516278
#kp, ki, kd = 2.88977477, 8.34949405, 0.01294233

for i in time:
    N_pid_arr.append(X_pid[1][0])
    
    sp = ref(i)
    ref_pid_arr.append(sp)
    
    u = pid_ctrl.compute(sp, X_pid[1][0], i, kp, ki, kd)
    U_pid_arr.append( u)
    
    X_pid = system.motor(X_pid, u) + np.random.normal(0,.1)

    if i % 5 <= 1e-2:
        X_pid[1][0] += np.random.randint(0, 1)
        
        
fpid_cfg = FopidConfig()
fpid_cfg.u_max = 24
fpid_cfg.u_min = -24

fpid_cfg.Lambda = 0.5
fpid_cfg.Mu = 0.5

fpid_ctrl = FOPID(fpid_cfg)

X_fpid = np.array([
    [0],
    [0]
])

N_fpid_arr = []
U_fpid_arr = []
ref_fpid_arr = []


bounds = [
    (1, 5),
    (1, 1),
    (0, 1e-1),
    (1e-6, 1),
    (1e-6, 1)
]

params = FOPID_tuner(system=system.motor, time= time, config=fpid_cfg, bounds=bounds, allowNoise=False, allowDist=False)
Kp, Ki, Kd, Lam, Mu = params
print(params)


fpid_cfg.Lambda = Lam #2.20053269e-01
fpid_cfg.Mu =  Mu #1.00000000e-06

#Kp, Ki, Kd = 8.50000000e+00, 4.0000000e+00, 1.00000000e-02


for i in time:
    N_fpid_arr.append(X_fpid[1][0])
    
    sp = ref(i)
    ref_fpid_arr.append(sp)
    
    u, v = fpid_ctrl.compute(sp, X_fpid[1][0], i, Kp, Ki, Kd)
    U_fpid_arr.append(u)
    
    X_fpid = system.motor(X_fpid, u) + np.random.normal(0,.1)

    if i % 5 <= 1e-2:
        X_fpid[1][0] += np.random.randint(0, 1)
        

plt.plot(time, N_pid_arr, label = "PID")     
plt.plot(time, N_fpid_arr, label = "FOPID")
plt.plot(time, ref_pid_arr, label = "reference")

plt.legend()
plt.show()
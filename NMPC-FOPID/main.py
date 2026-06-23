import mujoco
import mujoco.viewer
import numpy as np
from Controllers import PidConfig, PID, NMPC_mobileRobot


model = mujoco.MjModel.from_xml_path(
    "/home/alen/Codes/vscode/PythonCode/Main Project/amrModel2.xml"
)
data = mujoco.MjData(model)



nmpc = NMPC_mobileRobot(N=20, Ts=0.05)

pid_config = PidConfig(u_min=-50, u_max=50)
pid_L = PID(pid_config)
pid_R = PID(pid_config)



r = 0.1
b = 0.425



def joint_vel(name):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    addr = model.jnt_dofadr[jid]
    return data.qvel[addr]


def quat_to_yaw(q):
    w, x, y, z = q
    return np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))



dt = model.opt.timestep
sim_time = 0.0

with mujoco.viewer.launch_passive(model, data) as viewer:

    while viewer.is_running():


        pos  = data.qpos[0:3]
        quat = data.qpos[3:7]

        x = pos[0]
        y = pos[1]
        theta = quat_to_yaw(quat)

        state = np.array([x, y, theta])



        xr = np.array([2.0, 2.0, 0.0])

        x_next, u = nmpc.solveMPC(xr, state)

        v = u[0]
        omega = u[1]

        wL_des = (v - 0.5 * b * omega) / r
        wR_des = (v + 0.5 * b * omega) / r


        w_fl = joint_vel("joint_fl")
        w_rl = joint_vel("joint_rl")
        w_fr = joint_vel("joint_fr")
        w_rr = joint_vel("joint_rr")

        wL = 0.5 * (w_fl + w_rl)
        wR = 0.5 * (w_fr + w_rr)


        tau_L = pid_L.compute(
            setpoint=wL_des,
            measurement=wL,
            time=sim_time,
            Kp=1.0,
            Ki=2e-1,
            Kd=1e-6
        )

        tau_R = pid_R.compute(
            setpoint=wR_des,
            measurement=wR,
            time=sim_time,
            Kp=1.0,
            Ki=2e-1,
            Kd=1e-6
        )


        # -------------------------
        # 6. Apply torques
        # -------------------------
        data.ctrl[0] = tau_L
        data.ctrl[1] = tau_L
        data.ctrl[2] = tau_R
        data.ctrl[3] = tau_R


        # -------------------------
        # 7. Step simulation
        # -------------------------
        mujoco.mj_step(model, data)

        sim_time += dt

        viewer.sync()
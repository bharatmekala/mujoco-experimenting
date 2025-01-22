import mujoco
import mujoco.viewer
import numpy as np
import time

# Integration timestep in seconds. This corresponds to the amount of time the joint
# velocities will be integrated for to obtain the desired joint positions.
integration_dt: float = 0.1

# Damping term for the pseudoinverse. This is used to prevent joint velocities from
# becoming too large when the Jacobian is close to singular.
damping: float = 1e-4

# Gains for the twist computation. These should be between 0 and 1. 0 means no
# movement, 1 means move the end-effector to the target in one integration step.
Kpos: float = 0.95
Kori: float = 0.95

# Whether to enable gravity compensation.
gravity_compensation: bool = True

# Simulation timestep in seconds.
dt: float = 0.002

# Nullspace P gain.
Kn = np.asarray([10.0, 10.0, 10.0, 10.0, 5.0, 5.0, 5.0])

# Maximum allowable joint velocity in rad/s.
max_angvel = 0.785

 # Define gripper states
GRIPPER_OPEN = 0.0
GRIPPER_CLOSE = 220.0



def generate_random_targets():
    import numpy as np
    # Generate pos1 with x and y outside [-0.2, 0.2]
    pos1 = np.array([
        np.random.choice([np.random.uniform(-0.5, -0.2), np.random.uniform(0.2, 0.5)]),
        np.random.choice([np.random.uniform(-0.5, -0.2), np.random.uniform(0.2, 0.5)]),
        np.random.uniform(0.1, 1)
    ])
    
    # Scale pos1 to create pos2 with random scale factors for x and y
    scale_factor_x = np.random.uniform(1.0, 2)
    scale_factor_y = np.random.uniform(1.0, 2)
    pos2_x = pos1[0] * scale_factor_x
    pos2_y = pos1[1] * scale_factor_y
    
    # Clamp the x and y to remain within [-0.5, 0.5]
    pos2_x = np.clip(pos2_x, -1, 1)
    pos2_y = np.clip(pos2_y, -1, 1)
    
    pos2 = np.array([
        pos2_x,
        pos2_y,
        np.random.uniform(0.1, 1)
    ])
    
    return [
        {"pos": pos1, "quat": np.array([0.0, 1.0, 0.0, 0.0])},
        {"pos": pos2, "quat": np.array([0.0, 1.0, 0.0, 0.0])},
    ]



def main() -> None:
    
    targets = generate_random_targets()
    # Load the model and data
    model = mujoco.MjModel.from_xml_path("franka_emika_panda/scene.xml")
    data = mujoco.MjData(model)

    # Enable gravity compensation
    model.body_gravcomp[:] = float(gravity_compensation)
    model.opt.timestep = dt

    # End-effector site we wish to control
    site_name = "target_site"
    site_id = model.site(site_name).id
    
    # Define gripper actuator name
    gripper_act_name = "actuator8"
    
    # Get gripper actuator ID
    gripper_act_id = model.actuator(gripper_act_name).id
    
    
    #initialize the current target
    current_target = 0

    # Get the DOF and actuator IDs for the joints we wish to control
    joint_names = [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "joint7"
    ]
    
    act_names = [
        "actuator1",
        "actuator2",
        "actuator3",
        "actuator4",
        "actuator5",
        "actuator6",
        "actuator7"
    ]
    dof_ids = np.array([model.joint(name).id for name in joint_names])
    print(dof_ids)
    actuator_ids = np.array([model.actuator(name).id for name in act_names])

    # Initial joint configuration saved as a keyframe in the XML file
    key_name = "home"
    key_id = model.key(key_name).id
    q0 = model.key(key_name).qpos

    # Pre-allocate numpy arrays
    jac = np.zeros((6, model.nv))
    diag = damping * np.eye(6)
    eye = np.eye(model.nv)
    twist = np.zeros(6)
    site_quat = np.zeros(4)
    site_quat_conj = np.zeros(4)
    error_quat = np.zeros(4)

    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        # Reset the simulation
        mujoco.mj_resetDataKeyframe(model, data, key_id)

        # Reset the free camera
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Enable site frame visualization
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

        while viewer.is_running():
            step_start = time.time()
            
            # Current target
            target = targets[current_target]
            target_pos = target["pos"]
            target_quat = target["quat"]

            # Spatial velocity (aka twist)
            dx = target_pos - data.site(site_id).xpos
            twist[:3] = Kpos * dx / integration_dt
            mujoco.mju_mat2Quat(site_quat, data.site(site_id).xmat)
            mujoco.mju_negQuat(site_quat_conj, site_quat)
            mujoco.mju_mulQuat(error_quat, target_quat, site_quat_conj)
            mujoco.mju_quat2Vel(twist[3:], error_quat, 1.0)
            twist[3:] *= Kori / integration_dt
            
            if np.linalg.norm(dx) < 0.02 and np.linalg.norm(error_quat[1:]) < 0.02:
                print("Target reached")
                if current_target == len(targets) - 1:
                    print("Last target reached. Exiting.")
                    break
                else:
                    current_target += 1
            else:
                print(f"dx norm: {np.linalg.norm(dx)}, error_quat norm: {np.linalg.norm(error_quat[1:])}")

            # Jacobian
            mujoco.mj_jacSite(model, data, jac[:3], jac[3:], site_id)

            # Damped least squares
            dq = jac.T @ np.linalg.solve(jac @ jac.T + diag, twist)

            # Nullspace control biasing joint velocities towards the home configuration
            bias = np.zeros(model.nv)
            print(Kn.shape)
            print(q0[dof_ids].shape)
            bias[dof_ids] = Kn * (q0[dof_ids]- data.qpos[dof_ids])
            dq += (eye - np.linalg.pinv(jac) @ jac) @ bias

            # Clamp maximum joint velocity
            dq_abs_max = np.abs(dq).max()
            if dq_abs_max > max_angvel:
                dq *= max_angvel / dq_abs_max

            # Integrate joint velocities to obtain joint positions
            q = data.qpos.copy()  # Note the copy here is important
            mujoco.mj_integratePos(model, q, dq, integration_dt)
            # np.clip(q, *model.jnt_range.T, out=q)

            # Set the control signal and step the simulation
            data.ctrl[actuator_ids] = q[dof_ids]
            mujoco.mj_step(model, data)
            

            viewer.sync()
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()

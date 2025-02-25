import mujoco
import mujoco.viewer
import numpy as np
import time
import pickle  # add at the top along with other imports if not already present

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

    # Generate qpos for target[0] based on control ranges defined in panda.xml
    qpos_ranges = [
        (-0.5, 0.5),       # actuator1 (default range)
        (-1.7628, 1.7628), # actuator2 ctrlrange
        (-0.5, 0.5),       # actuator3 (default range)
        (-3.0718, -0.0698),# actuator4 ctrlrange
        (-0.5, 0.5),       # actuator5 (default range)
        (-0.0175, 3.7525), # actuator6 ctrlrange
        (-0.5, 0.5),       # actuator7 (default range)
        (0, 255),          # actuator8 ctrlrange (for tendon "split")
        (-0.5, 0.5)        # extra DOF (default)
    ]
    qpos0 = np.array([np.random.uniform(low, high) for (low, high) in qpos_ranges])
    target0 = {
        "qpos": qpos0,
        "quat": np.array([0.0, 0.0, 0.0, 0.0])
    }
    
    # Generate pos for target[1] with x and y outside [-0.2, 0.2]
    pos1 = np.array([
        np.random.choice([np.random.uniform(-0.5, -0.2), np.random.uniform(0.2, 0.5)]),
        np.random.choice([np.random.uniform(-0.5, -0.2), np.random.uniform(0.2, 0.5)]),
        np.random.uniform(0.3, 0.7)
    ])
    
    # Scale pos1 to create pos2 with random scale factors for x and y
    scale_factor_x = np.random.uniform(0.75, 1.25)
    scale_factor_y = np.random.uniform(0.75, 1.25)
    pos2_x = np.clip(pos1[0] * scale_factor_x, -0.5, 0.5)
    pos2_y = np.clip(pos1[1] * scale_factor_y, -0.5, 0.5)
    
    pos2 = np.array([
        pos2_x,
        pos2_y,
        np.random.uniform(0.3, 0.5)
    ])
    
    target1 = {"pos": pos2, "quat": np.array([0.0, 0.0, 0.0, 0.0])}
    
    return [target0, target1]

def save_demo_data_v2(qpos_list, joint_names, dof_ids, home_q, gripper_open):
    """
    Saves collected trajectories into a pickle file (demo_v2.pkl) in data format v2.
    """
    actions = []
    for qpos in qpos_list:
        dof_pos = {joint: float(qpos[dof_ids[i]]) for i, joint in enumerate(joint_names)}
        action = {
            "dof_pos_target": dof_pos,
            "ee_pose_target": {
                "pos": [0.0, 0.0, 0.0],
                "rot": [1.0, 0.0, 0.0, 0.0],
                "gripper_joint_pos": gripper_open
            }
        }
        actions.append(action)
    
    if qpos_list:
        init_dof = {joint: float(qpos_list[0][dof_ids[i]]) for i, joint in enumerate(joint_names)}
    else:
        init_dof = {joint: float(home_q[dof_ids[i]]) for i, joint in enumerate(joint_names)}
    
    init_state = {
        "franka": {
            "pos": [0.0, 0.0, 0.0],
            "rot": [1.0, 0.0, 0.0, 0.0],
            "dof_pos": init_dof
        }
    }
    
    states = []
    for qpos in qpos_list:
        state_dof = {joint: float(qpos[dof_ids[i]]) for i, joint in enumerate(joint_names)}
        state = {
            "franka": {
                "pos": [0.0, 0.0, 0.0],
                "rot": [1.0, 0.0, 0.0, 0.0],
                "dof_pos": state_dof
            }
        }
        states.append(state)
    
    demo = {
        "actions": actions,
        "init_state": init_state,
        "states": states,
        "extra": None
    }
    demo_data = {"franka": [demo]}
    
    with open("demo_v2.pkl", "wb") as f:
        pickle.dump(demo_data, f)
    print("Saved demonstration data to demo_v2.pkl")

def main() -> None:
    
    targets = generate_random_targets()
    # Load the model and data
    model = mujoco.MjModel.from_xml_path("franka_emika_panda/scene.xml")
    data = mujoco.MjData(model)
    
    data.qpos[:] = np.array(targets[0]["qpos"])
    data.ctrl[:] = np.array(targets[0]["qpos"][:8])

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
    current_target = 1

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

    # Initialize list to record qpos
    qpos_list = []

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

            # Record qpos after the first target is reached
            if current_target >= 1:
                qpos_list.append(data.qpos.copy())

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
            
            #save the data
            #qpos_list.append(data.qpos.copy())
            #ctrl_list.append(data.ctrl.copy())
            #init_controller(model, data)

            viewer.sync()
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
        
        # Save qpos_list to data.txt after simulation ends
        with open("data.txt", "w") as file:
            for qpos in qpos_list:
                qpos_str = ' '.join(map(str, qpos))
                file.write(f"{qpos_str}\n")
    
    # Call new function to save demonstration data in v2 format
    save_demo_data_v2(qpos_list, joint_names, dof_ids, q0, GRIPPER_OPEN)

if __name__ == "__main__":
    main()

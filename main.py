import mujoco
import glfw  # Required for visualization

# Initialize GLFW (graphics library for visualization)
glfw.init()

# Load the model
model = mujoco.MjModel.from_xml_path("robot/gen3.xml")  # Replace with your XML file
data = mujoco.MjData(model)

# Create a GLFW window
window = mujoco.MjvGLContext()

# Set up the scene and camera
scn = mujoco.MjvScene(model, maxgeom=1000)
cam = mujoco.MjvCamera()

# Render and simulate
while not glfw.window_should_close(window):
    mujoco.mj_step(model, data)
    mujoco.mjv_updateScene(model, data, scn, cam, None)
    glfw.swap_buffers(window)
    glfw.poll_events()

# Clean up
glfw.terminate()

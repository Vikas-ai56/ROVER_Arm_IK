# Since package_dir={"": "include"}, modules are imported directly
# Not as arm_ik.module but just as module
from configuration import Configuration
from solve_ik import solve_ik
from tasks import FrameTask, PostureTask, DampingTask
from limits import ConfigurationLimit, VelocityLimit
from lie import SE3, SO3

print("✅ All imports successful!")
print(f"Configuration: {Configuration}")
print(f"SE3: {SE3}")


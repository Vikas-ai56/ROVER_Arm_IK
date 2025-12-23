import numpy as np
import pinocchio
from numpy.linalg import norm, solve
from pathlib import Path
import math
import mujoco
import mujoco.viewer
import time

from viz_test import URDF_STRING

URDF_PATH = Path(__file__).parent / "rover_astra" / "src" / "roverArm" / "rover_arm" / "urdf" / "rover_arm_fixed.urdf"
MESH_DIR = Path(__file__).parent / "rover_astra" / "src" / "roverArm" / "rover_arm" / "meshes"

model = pinocchio.buildModelFromUrdf(str(URDF_PATH))
data = model.createData()
 
JOINT_ID = 6
oMdes = pinocchio.SE3(np.eye(3), np.array([0.376, 0.023, 0.363]))
 
q = pinocchio.neutral(model)
eps = 1e-4
IT_MAX = 1000
DT = 1e-1
damp = 1e-12
 
i = 0
while True:
    pinocchio.forwardKinematics(model, data, q)
    iMd = data.oMi[JOINT_ID].actInv(oMdes)
    err = pinocchio.log(iMd).vector  # in joint frame
    if norm(err) < eps:
        success = True
        break
    if i >= IT_MAX:
        success = False
        break
    J = pinocchio.computeJointJacobian(model, data, q, JOINT_ID)  # in joint frame
    J = -np.dot(pinocchio.Jlog6(iMd.inverse()), J)
    v = -J.T.dot(solve(J.dot(J.T) + damp * np.eye(6), err))
    q = pinocchio.integrate(model, q, v * DT)
    if not i % 10:
        print(f"{i}: error = {err.T}")
    i += 1
 
if success:
    print("\n" + "="*50)
    print("Convergence achieved!")
    print("="*50)
    
    # Print joint angles with names
    print("\nFinal Joint Angles:")
    print("-"*40)
    for i in range(1, model.njoints):  # Skip index 0 (universe/root)
        joint_name = model.names[i]
        angle_rad = q[i-1] % 2*math.pi  # q is 0-indexed for actual joints
        angle_deg = np.degrees(angle_rad) % 360
        print(f"  {joint_name:20s}: {angle_rad:+.4f} rad  ({angle_deg:+.2f}°)")
    print("-"*40)
else:
    print(
        "\n"
        "Warning: the iterative algorithm has not reached convergence "
        "to the desired precision"
    )

print(f"\nFinal position error: {norm(err[:3])*1000:.3f} mm")
print(f"Final orientation error: {norm(err[3:])*1000:.3f} mrad")
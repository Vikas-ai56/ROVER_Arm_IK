# ----------------------------------------------------
# Keyboard controller
# ----------------------------------------------------

import numpy as np
import pinocchio
from numpy.linalg import norm, solve
from pathlib import Path
import mujoco
import mujoco.viewer
import time
import threading
import msvcrt

from viz_test import URDF_STRING

URDF_PATH = Path(__file__).parent / "rover_astra" / "src" / "roverArm" / "rover_arm" / "urdf" / "rover_arm_fixed.urdf"
MESH_DIR = Path(__file__).parent / "rover_astra" / "src" / "roverArm" / "rover_arm" / "meshes"


EPS = 1e-4
IT_MAX = 1000
DT = 1e-1
DAMP = 1e-12
JOINT_ID = 6

class DifferentialIK:
    def __init__(self, ee_frame: str):
        self.model = pinocchio.buildModelFromUrdf(str(URDF_PATH))
        self.data = self.model.createData()
        self.q = None
        self.frame_id = self.model.getFrameId(ee_frame)

        self.target_pos = None
        self.running = True
        self.solving = False
        self.lock = threading.Lock()
        #initialize
        self.initialize()

    def initialize(self):
        self.q = pinocchio.neutral(self.model)
        pinocchio.framesForwardKinematics(self.model, self.data, self.q)
        self.target_pos = self.data.oMf[self.frame_id].translation.copy()
        
        assets = {p.name: p.read_bytes() for p in MESH_DIR.glob("*.STL")}
        self.mj_model = mujoco.MjModel.from_xml_string(URDF_STRING, assets=assets)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.mj_data.qpos[:] = self.q
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def solve_ik(self, target: np.ndarray) -> bool:
        oMdes = pinocchio.SE3(np.eye(3), target)
        q = self.q.copy()
        
        for i in range(IT_MAX):
            pinocchio.forwardKinematics(self.model, self.data, q)
            iMd = self.data.oMi[JOINT_ID].actInv(oMdes)
            err = pinocchio.log(iMd).vector
            
            if norm(err) < EPS:
# ---------------------------------------------------
# Changes made
# ---------------------------------------------------
                self.q = np.clip(q, self.model.lowerPositionLimit + 0.01, 
                                self.model.upperPositionLimit - 0.01)
                return True
            
            J = pinocchio.computeJointJacobian(self.model, self.data, q, JOINT_ID)
            J = -np.dot(pinocchio.Jlog6(iMd.inverse()), J)
            v = -J.T.dot(solve(J.dot(J.T) + DAMP * np.eye(6), err))
            q = pinocchio.integrate(self.model, q, v * DT)

# ------------------------------------------------------
# Added the line below
# ------------------------------------------------------
            q = np.clip(q, self.model.lowerPositionLimit + 0.01, 
                       self.model.upperPositionLimit - 0.01)
        
        self.q = q        
        return False
    
    def set_target(self, x: float, y: float, z: float):
        with self.lock:
            self.target_pos = np.array([x, y, z])
            self.solving = True
    
    def update(self):
        with self.lock:
            if not self.solving or self.target_pos is None:
                return
            
            success = self.solve_ik(self.target_pos)
            
            pinocchio.forwardKinematics(self.model, self.data, self.q)
            actual = self.data.oMi[JOINT_ID].translation
            
            status = "✓" if success else "✗"
            print(f"{status} Target: {self.target_pos} | Actual: {actual.round(4)} | Joints(deg): {np.degrees(self.q).round(1)}")
            self.solving = False

    def _displace(self, isForward):
        pinocchio.forwardKinematics(self.model, self.data, self.q)
        ee_rotation = self.data.oMi[JOINT_ID].rotation
        # Forward direction is X-axis of EE frame (first column)
        forward_dir = ee_rotation[:, 0]
        # Project onto XY plane and normalize
        forward_xy = forward_dir[:2]
        forward_xy = forward_xy / (norm(forward_xy) + 1e-6)
        
        step = 0.01 if isForward else -0.01
        x = self.target_pos[0] + forward_xy[0] * step
        y = self.target_pos[1] + forward_xy[1] * step
        return (x, y, self.target_pos[2])
    
    def _rotate_target_around_base(self, angle_rad: float = np.radians(5)):
        x, y = self.target_pos[0], self.target_pos[1]
        
        current_angle = np.arctan2(y, x)
        radius = np.sqrt(x**2 + y**2)
        
        new_angle = current_angle + angle_rad        
        return (radius * np.cos(new_angle), radius * np.sin(new_angle), self.target_pos[2])
    
    def keyboard_input(self):
        if self.running:
            x, y, z = self.target_pos
            if msvcrt.kbhit():
                key = msvcrt.getch().decode('utf-8').lower()
                if key == 'w': x,y,z = self._displace(True)
                elif key == 's': x,y,z = self._displace(False)
                elif key == 'a': x,y,z = self._rotate_target_around_base(np.radians(5))
                elif key == 'd': x,y,z = self._rotate_target_around_base(np.radians(-5))
                elif key == 'q': z = z + 0.01
                elif key == 'e': z = z - 0.01
            return (x,y,z)

    def input_loop(self):
        print("WASD: move | QE: up/down | X: quit")
        while self.running:
            try:
                if msvcrt.kbhit():
                    key = msvcrt.getch().decode('utf-8').lower()
                    x, y, z = self.target_pos
                    if key == 'w': x,y,z = self._displace(True)
                    elif key == 's': x,y,z = self._displace(False)
                    elif key == 'a': x,y,z = self._rotate_target_around_base(np.radians(5))
                    elif key == 'd': x,y,z = self._rotate_target_around_base(np.radians(-5))
                    elif key == 'q': z = z + 0.1
                    elif key == 'e': z = z - 0.1
                    elif key == 'x':
                        self.running = False
                        break
                    else:
                        continue
                    self.set_target(x, y, z)
                    self.update()
                time.sleep(0.01)
            except Exception as e:
                print(f"[ERROR in input_loop]: {e}")

    def run(self):
        with mujoco.viewer.launch_passive(self.mj_model, self.mj_data) as viewer:
            while self.running and viewer.is_running():
                self.mj_data.qpos[:] = self.q
                mujoco.mj_forward(self.mj_model, self.mj_data)
                viewer.sync()
                time.sleep(0.01)
                
def main():
    ctrl = DifferentialIK(ee_frame="ee_link")
    t = threading.Thread(target=ctrl.input_loop, daemon=True)
    t.start()
    ctrl.run()

if __name__ == "__main__":
    main()


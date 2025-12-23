import sys
from pathlib import Path
import numpy as np
import time
import threading
import msvcrt 

# --- Project Imports ---
sys.path.insert(0, str(Path(__file__).parent.parent))
from include.solve_ik import solve_ik
from include.tasks import FrameTask, PostureTask
from include.limits import VelocityLimit, ConfigurationLimit
from include.lie import SE3, SO3
from include.configuration import Configuration
from viz_test import URDF_STRING
import pinocchio as pin
import mujoco
import mujoco.viewer

# --- Constants ---
URDF_PATH = Path(__file__).parent.parent / "rover_astra" / "src" / "roverArm" / "rover_arm" / "urdf" / "rover_arm_fixed.urdf"
MESH_DIR = Path(__file__).parent.parent / "rover_astra" / "src" / "roverArm" / "rover_arm" / "meshes"

# 1. TIGHTER SPEED LIMITS (Rad/s)
# Real robots rarely move faster than 0.5 rad/s during teleop
VELOCITY_LIMITS = {
    "shoulder_joint": 1.0, 
    "upperarm_joint": 1.0, 
    "elbow_joint": 1.0,
    "wrist1_joint": 1.0, 
    "wrist2_joint": 1.0
}

class TeleopController:
    def __init__(self, ee_name: str, safety_break: bool = False):
        self.safety_break = safety_break
        self.dt = 0.01  # 100 Hz
        
        self.step_size = 0.01 

        self.model = pin.buildModelFromUrdf(str(URDF_PATH))
        self.config = Configuration(self.model)
        self.ee_frame = ee_name

        q_init = (self.model.lowerPositionLimit + self.model.upperPositionLimit) / 2.0

        q_clamped = np.clip(
            q_init,
            self.model.lowerPositionLimit + 0.1,
            self.model.upperPositionLimit - 0.1
        )
        self.config.update(q_clamped)

        self.frame_task = FrameTask(
            frame_name=ee_name,
            position_cost=10.0,    # HIGH - reaching target is priority
            orientation_cost=0.6,  # LOW - we only care about position
            gain=1.0,              # Full tracking - no lag
            lm_damping=1e-3        # Low damping for responsive motion
        )
        
        # Very low posture cost - only for numerical stability, not motion control
        self.posture_task = PostureTask(
            model=self.model,
            cost=1e-3  # Tiny - just prevents rank deficiency
        )

        self.limits = [
            ConfigurationLimit(model=self.model, gain=0.85), 
            VelocityLimit(model=self.model, velocities=VELOCITY_LIMITS)
        ]
        self.tasks = [self.frame_task, self.posture_task]

        # 4. VELOCITY SMOOTHING STATE
        self.prev_velocity = np.zeros(self.model.nv)
        self.smoothing_alpha = 0.7 # 70% new command, 30% momentum (More responsive)

        # State Init
        self.target_rot = np.eye(3)
        self.running = True
        self.action_lock = threading.Lock()
        self.last_print_time = 0
        self.active_command = False  
        
        self.prev_dist = float('inf')
        self.stall_count = 0
        self.max_stall_count = 250  

        # MuJoCo Init
        self.mj_model = None
        self.mj_data = None
        self.mujoco_init()
                
        ee_transform = self.config.get_transform_frame_to_world(self.ee_frame)
        self.target_pos = ee_transform.translation.copy()
        print(f"Starting config (deg): {np.degrees(self.config.q)}")
        print(f"Starting EE position: {self.target_pos}")
        
    def mujoco_init(self):
        assets = {p.name: p.read_bytes() for p in MESH_DIR.glob("*.STL")}
        self.mj_model = mujoco.MjModel.from_xml_string(URDF_STRING, assets=assets)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.mj_data.qpos[:] = self.config.q
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def update_target_from_action(self, action: str):
        with self.action_lock:            
            if action == 'forward':
                self._displace(True)
            elif action == 'backward':
                self._displace(False)
            elif action == 'left':
                self._rotate_target_around_base(np.radians(5))
            elif action == 'right':
                self._rotate_target_around_base(np.radians(-5))
            elif action == 'up':
                self.target_pos[2] += self.step_size
            elif action == 'down':
                self.target_pos[2] -= self.step_size
            elif action == 'stop':
                curr_ee = self.config.get_transform_frame_to_world(self.ee_frame)
                self.target_pos = curr_ee.translation.copy()
                self.active_command = False
                print("⏹ Stopped")
                return
            
            # Activate motion
            self.active_command = True
            self.stall_count = 0
            self.prev_dist = float('inf')

    def _displace (self, isForward):
        current_angle = np.arctan2(self.target_pos[1], self.target_pos[0])
        if isForward:
            self.target_pos[0] += np.cos(current_angle) * self.step_size
            self.target_pos[1] += np.sin(current_angle) * self.step_size
        else:
            self.target_pos[0] -= np.cos(current_angle) * self.step_size
            self.target_pos[1] -= np.sin(current_angle) * self.step_size
    
    def _rotate_target_around_base(self, angle_rad: float):
        x, y = self.target_pos[0], self.target_pos[1]
        
        current_angle = np.arctan2(y, x)
        radius = np.sqrt(x**2 + y**2)
        
        new_angle = current_angle + angle_rad
        
        self.target_pos[0] = radius * np.cos(new_angle)
        self.target_pos[1] = radius * np.sin(new_angle)
    
    def set_target(self, x: float, y: float, z: float):
        """Set absolute target position."""
        with self.action_lock:
            self.target_pos = np.array([x, y, z], dtype=float)
            self.target_rot = np.eye(3)
            self.active_command = True
            self.stall_count = 0
            self.prev_dist = float('inf')
        print(f"Target set: [{x:.3f}, {y:.3f}, {z:.3f}]")

    def control_step(self):
        with self.action_lock:
            is_active = self.active_command
            target_pos_copy = self.target_pos.copy()
            target_rot_copy = self.target_rot.copy()
        
        if not is_active:
            self.prev_velocity = np.zeros(self.model.nv)
            mujoco.mj_forward(self.mj_model, self.mj_data)
            return
        
        curr_ee = self.config.get_transform_frame_to_world(self.ee_frame)
        dist = np.linalg.norm(curr_ee.translation - target_pos_copy)
        
        if dist < 0.005:  # 5mm
            with self.action_lock:
                self.active_command = False
                self.target_pos = curr_ee.translation.copy()
            self.prev_velocity = np.zeros(self.model.nv)
            print(f"Reached target (dist: {dist*1000:.1f}mm)")
            return
        
        # STALL DETECTION: If not making progress, stop
        progress = self.prev_dist - dist
        if progress < 0.0001:  # Less than 0.1mm progress
            self.stall_count += 1
        else:
            self.stall_count = 0
        self.prev_dist = dist
        
        if self.stall_count > self.max_stall_count:
            with self.action_lock:
                self.active_command = False
                self.target_pos = curr_ee.translation.copy()
            self.prev_velocity = np.zeros(self.model.nv)
            print(f"Cannot reach target (dist: {dist*1000:.1f}mm)")
            return
        
        # Set target for IK
        target_se3 = SE3.from_rotation_and_translation(
            SO3.from_matrix(target_rot_copy), target_pos_copy
        )
        self.frame_task.set_target(target_se3)
        
        # Need to update the current joints config of the robot
        self.posture_task.set_target(self.config.q)
        
        # Solve IK
        try:
            raw_velocity = solve_ik(
                configuration=self.config,
                tasks=self.tasks,
                dt=self.dt,
                limits=self.limits,
                damping=1e-4  # Low damping = more accurate tracking
            )
        except Exception as e:
            print(f"⚠️ IK FAILED: {e}")
            raw_velocity = np.zeros(self.model.nv)

        velocity = (self.smoothing_alpha * raw_velocity) + \
                   ((1 - self.smoothing_alpha) * self.prev_velocity)
        velocity = np.clip(velocity, -0.5, 0.5)  
        self.prev_velocity = raw_velocity

# -------------------------------------------------------------------------------
# Update the joint configuration
# -------------------------------------------------------------------------------

        q_new = self.config.integrate(velocity, self.dt)
        q_clamped = np.clip(
            q_new,
            self.config.model.lowerPositionLimit + 0.01,
            self.config.model.upperPositionLimit - 0.01
        )
        
        self.config.update(q_clamped)
        self.mj_data.qpos[:] = q_clamped
        mujoco.mj_forward(self.mj_model, self.mj_data)

        if time.time() - self.last_print_time > 0.3:
            self.current_state(velocity, dist)
            self.last_print_time = time.time()

    def current_state(self, velocity, dist):
        curr_ee = self.config.get_transform_frame_to_world(self.ee_frame)
        angles_deg = np.degrees(self.config.q)
        
        tgt = self.target_pos
        cur = curr_ee.translation
        print(f"Target: [{tgt[0]:.4f}, {tgt[1]:.4f}, {tgt[2]:.4f}]")
        print(f"Actual: [{cur[0]:.4f}, {cur[1]:.4f}, {cur[2]:.4f}]")
        print(f"Dist: {dist*1000:.1f}mm | MaxVel: {np.max(np.abs(velocity)):.3f}")
        print(f"Joints: [{', '.join([f'{a:.1f}' for a in angles_deg])}]")
        print("-" * 50)

    def run(self):
        with mujoco.viewer.launch_passive(self.mj_model, self.mj_data) as viewer:
            while self.running and viewer.is_running():
                step_start = time.time()
                self.control_step()
                viewer.sync()
                
                elapsed = time.time() - step_start
                if elapsed < self.dt:
                    time.sleep(self.dt - elapsed)

    def stop(self):
        self.running = False
    
    def coordinate_input_loop(self):
        """Loop for entering absolute coordinates."""
        print("\n=== Coordinate Input Mode ===")
        print("Enter: x y z  OR  'k' for keyboard mode  OR  'q' to quit\n")
        
        while self.running:
            try:
                user_input = input("Coords (x y z): ").strip().lower()
                
                if user_input == 'q':
                    self.stop()
                    break
                elif user_input == 'k':
                    print("Switched to keyboard mode (WASD/QE)")
                    return 'keyboard'
                
                coords = user_input.replace(',', ' ').split()
                if len(coords) != 3:
                    print("Enter 3 values: x y z")
                    continue
                    
                x, y, z = float(coords[0]), float(coords[1]), float(coords[2])
                self.set_target(x, y, z)
                
                # Wait for motion to complete
                while self.active_command and self.running:
                    time.sleep(0.1)
                
                curr_ee = self.config.get_transform_frame_to_world(self.ee_frame)
                print(f"Reached: [{curr_ee.translation[0]:.3f}, {curr_ee.translation[1]:.3f}, {curr_ee.translation[2]:.3f}]\n")
                    
            except ValueError:
                print("Invalid format. Use: x y z")
            except EOFError:
                break
        return None

def main():
    controller = TeleopController(ee_name="ee_link")
    
    print("\n" + "="*50)
    print("  ROBOT ARM CONTROLLER")
    print("="*50)
    print("\nModes:")
    print("  [1] Keyboard teleop (WASD/QE)")
    print("  [2] Coordinate input (x y z)")
    print("  [x] Exit")
    print("="*50 + "\n")
    
    mode = 'keyboard'  # Start with keyboard mode
    
    def keyboard_teleop():
        nonlocal mode
        print("\n--- Keyboard Mode ---")
        print("W/S: Forward/Back | A/D: Left/Right | Q/E: Up/Down")
        print("Space: Stop | C: Coordinate mode | X: Exit\n")
        
        while controller.running and mode == 'keyboard':
            if msvcrt.kbhit():
                key = msvcrt.getch().decode('utf-8').lower()
                if key == 'w': controller.update_target_from_action('forward')
                elif key == 's': controller.update_target_from_action('backward')
                elif key == 'a': controller.update_target_from_action('left')
                elif key == 'd': controller.update_target_from_action('right')
                elif key == 'q': controller.update_target_from_action('up')
                elif key == 'e': controller.update_target_from_action('down')
                elif key == ' ': controller.update_target_from_action('stop')
                elif key == 'c': 
                    mode = 'coords'
                    print("\nSwitching to coordinate mode...")
                elif key == 'x': 
                    controller.stop()
                    break
                time.sleep(0.01)
            else:
                time.sleep(0.01)
    
    def input_handler():
        nonlocal mode
        while controller.running:
            if mode == 'keyboard':
                keyboard_teleop()
            elif mode == 'coords':
                result = controller.coordinate_input_loop()
                if result == 'keyboard':
                    mode = 'keyboard'
    
    # Start input handler thread
    t = threading.Thread(target=input_handler, daemon=True)
    t.start()
    
    # Run main control loop
    controller.run()

if __name__ == "__main__":
    main()
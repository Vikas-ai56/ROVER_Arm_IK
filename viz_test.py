import mujoco
import mujoco.viewer
import time
from pathlib import Path
MESH_DIR = Path(__file__).parent / "rover_astra" / "src" / "roverArm" / "rover_arm" / "meshes"
# MESH_DIR = Path(r"E:\Arm_IK\rover_astra\src\roverArm\rover_arm\meshes")

print(f"Loading meshes from: {MESH_DIR}")
assets = {}

try:
    for file_path in MESH_DIR.glob("*.STL"):
        with open(file_path, "rb") as f:
            assets[file_path.name] = f.read()
            print(f"  - Loaded asset: {file_path.name}")
except FileNotFoundError:
    print("CRITICAL ERROR: Could not find the mesh directory.")
    print(f"Checked path: {MESH_DIR}")
    exit()

if not assets:
    print("WARNING: No STL files found! Check the directory path.")

URDF_STRING = """
<?xml version='1.0' encoding='utf-8'?>
<robot name="rover_arm">
  
  <link name="world" />

  <joint name="world_joint" type="fixed">
    <parent link="world" />
    <child link="base_link" />
    <origin xyz="0 0 0" rpy="0 0 0" />
  </joint>

  <link name="base_link">
    <inertial>
      <origin xyz="0.00687930292992112 -5.41233724504764E-16 0.0134758677537272" rpy="0 0 0" />
      <mass value="0.107549490172523" />
      <inertia ixx="0.000129185873904102" ixy="1.35843292566697E-19" ixz="3.62338947266451E-19" iyy="0.000167948580816449" iyz="-1.18461477534074E-18" izz="0.000292377352216446" />
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry>
        <mesh filename="base_link.STL" />
      </geometry>
      <material name="">
        <color rgba="0.996078431372549 0.996078431372549 1 1" />
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry>
        <mesh filename="base_link.STL" />
      </geometry>
    </collision>
  </link>

  <link name="shoulder_link">
    <inertial>
      <origin xyz="8.32231213188095E-06 0.0203402526286956 0.0551734534765064" rpy="0 0 0" />
      <mass value="0.855475627966906" />
      <inertia ixx="0.00111103760161046" ixy="1.94540837133742E-08" ixz="2.1295728671634E-08" iyy="0.00127941202345426" iyz="5.19186552494487E-07" izz="0.00084308013698595" />
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry>
        <mesh filename="shoulder_link.STL" />
      </geometry>
      <material name="">
        <color rgba="0.96078431372549 0.96078431372549 0.964705882352941 1" />
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry>
        <mesh filename="shoulder_link.STL" />
      </geometry>
    </collision>
  </link>

  <joint name="shoulder_joint" type="revolute">
    <origin xyz="0 0 0.0345" rpy="0 0 0" />
    <parent link="base_link" />
    <child link="shoulder_link" />
    <axis xyz="0 0 1" />
    <limit lower="-3.14" upper="3.14" effort="10.5" velocity="1.5" />
  </joint>

  <link name="upperarm_link">
    <inertial>
      <origin xyz="-2.70918179823276E-06 0.034430199275424 0.310830598763632" rpy="0 0 0" />
      <mass value="1.69162786755335" />
      <inertia ixx="0.0101293308172892" ixy="-5.91623806826241E-08" ixz="-9.28412403243426E-08" iyy="0.0105489254219991" iyz="-4.30319037658778E-07" izz="0.000907812411984246" />
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry>
        <mesh filename="upperarm_link.STL" />
      </geometry>
      <material name="">
        <color rgba="0.266666666666667 0.588235294117647 0.282352941176471 1" />
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry>
        <mesh filename="upperarm_link.STL" />
      </geometry>
    </collision>
  </link>

  <joint name="upperarm_joint" type="revolute">
    <origin xyz="0 -0.044706 0.083" rpy="0 0 0" />
    <parent link="shoulder_link" />
    <child link="upperarm_link" />
    <axis xyz="0 -1 0" />
    <limit lower="-1.57" upper="1.57" effort="10.5" velocity="1.5" />
  </joint>

  <link name="elbow_link">
    <inertial>
      <origin xyz="0.204722227074469 -0.0419483035815702 -0.00157486053082279" rpy="0 0 0" />
      <mass value="1.07922630503039" />
      <inertia ixx="0.000419181010760536" ixy="7.11741809370692E-08" ixz="6.10025801738287E-06" iyy="0.00391113124957561" iyz="-3.06017160267862E-12" izz="0.00358543795771881" />
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry>
        <mesh filename="elbow_link.STL" />
      </geometry>
      <material name="">
        <color rgba="0.266666666666667 0.588235294117647 0.282352941176471 1" />
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry>
        <mesh filename="elbow_link.STL" />
      </geometry>
    </collision>
  </link>

  <joint name="elbow_joint" type="revolute">
    <origin xyz="0 0.086169 0.497" rpy="0 0 0" />
    <parent link="upperarm_link" />
    <child link="elbow_link" />
    <axis xyz="0 -1 0" />
    <limit lower="-1.57" upper="3.14" effort="10.5" velocity="1.5" />
  </joint>

  <link name="wrist1_link">
    <inertial>
      <origin xyz="0.0302374140928486 -0.0029999997213056 5.0144521868134E-10" rpy="0 0 0" />
      <mass value="0.039237317766174" />
      <inertia ixx="1.15363222488905E-05" ixy="4.01230758868372E-14" ixz="8.39085802463508E-14" iyy="6.67525690136622E-06" iyz="-7.70014150865565E-14" izz="6.60545008536105E-06" />
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry>
        <mesh filename="wrist1_link.STL" />
      </geometry>
      <material name="">
        <color rgba="0.792156862745098 0.819607843137255 0.933333333333333 1" />
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry>
        <mesh filename="wrist1_link.STL" />
      </geometry>
    </collision>
  </link>

  <joint name="wrist1_joint" type="revolute">
    <origin xyz="0.40386 -0.039 -0.011" rpy="0 0 0" />
    <parent link="elbow_link" />
    <child link="wrist1_link" />
    <axis xyz="0 1 0" />
    <limit lower="-1.57" upper="1.57" effort="10.5" velocity="1.5" />
  </joint>

  <link name="wrist2_link">
    <inertial>
      <origin xyz="0.0412194856485808 -0.0161936654386357 7.08549571259987E-06" rpy="0 0 0" />
      <mass value="0.132797730298268" />
      <inertia ixx="0.000325802306284526" ixy="0.000112637927863482" ixz="1.47176339144941E-08" iyy="0.000208154553092543" iyz="-1.50492483398845E-08" izz="0.000516192730665238" />
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry>
        <mesh filename="wrist2_link.STL" />
      </geometry>
      <material name="">
        <color rgba="0.792156862745098 0.819607843137255 0.933333333333333 1" />
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry>
        <mesh filename="wrist2_link.STL" />
      </geometry>
    </collision>
  </link>

  <joint name="wrist2_joint" type="revolute">
    <origin xyz="0.049 -0.003 0" rpy="0 0 0" />
    <parent link="wrist1_link" />
    <child link="wrist2_link" />
    <axis xyz="-1 0 0" />
    <limit lower="-3.14" upper="3.14" effort="10.5" velocity="1.5" />
  </joint>

  <link name="ee_link">
    <inertial>
      <origin xyz="-0.045802 0.0041133 -0.00845" rpy="0 0 0" />
      <mass value="0.054422" />
      <inertia ixx="4.2801E-06" ixy="2.2691E-06" ixz="1.4323E-07" iyy="6.8002E-05" iyz="-3.0715E-09" izz="6.801E-05" />
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry>
        <mesh filename="ee_link.STL" />
      </geometry>
      <material name="">
        <color rgba="0 0 0 1" />
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry>
        <mesh filename="ee_link.STL" />
      </geometry>
    </collision>
  </link>

  <joint name="ee_joint" type="fixed">
    <origin xyz="0.13 0.002 0.01" rpy="0 0 0" />
    <parent link="wrist2_link" />
    <child link="ee_link" />
    <axis xyz="0 0 0" />
  </joint>

  <link name="ee_rotation_z">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0" />
      <mass value="0.001" />
      <inertia ixx="1e-6" ixy="0" ixz="0" iyy="1e-6" iyz="0" izz="1e-6" />
    </inertial>
  </link>

  <joint name="ee_rotation_joint_z" type="revolute">
    <origin xyz="0 0 0" rpy="0 0 0" />
    <parent link="ee_link" />
    <child link="ee_rotation_z" />
    <axis xyz="0 0 1" />
    <limit lower="-3.1416" upper="3.1416" effort="10.5" velocity="1.5" />
  </joint>

</robot>
"""

# =============================================================================
# 3. MAIN LOOP
# =============================================================================
def main():
    print("Attempting to load URDF...")
    
    try:
        model = mujoco.MjModel.from_xml_string(URDF_STRING, assets=assets)
        data = mujoco.MjData(model)
        print("SUCCESS! Model loaded.")
    except ValueError as e:
        print("\n-------------------------------------------------------------")
        print("MUJOCO ERROR DETECTED:")
        print(e)
        if "inertia" in str(e) or "mass" in str(e):
             print("\n!!! CRITICAL WARNING !!!")
             print("You requested to ignore the inertia modification for 'ee_rotation_z'.")
             print("However, MuJoCo REQUIRES mass/inertia for all moving bodies.")
             print("To fix this, you must add the <inertial> block to the last link.")
        print("-------------------------------------------------------------\n")
        return
    except Exception as e:
        print(f"FAILED: {e}")
        return

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start = time.time()
        print("Viewer launched. Closing in 30 seconds...")
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.01)

if __name__ == "__main__":
    main()
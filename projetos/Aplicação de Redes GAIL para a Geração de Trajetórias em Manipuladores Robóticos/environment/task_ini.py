from pyrcareworld.envs.dressing_env import DressingEnv
import pyrcareworld.attributes as attr
import cv2
import numpy as np
from numpy.typing import ArrayLike
from pyrcareworld.attributes import CameraAttr

intrinsics = np.eye(3)

intrinsics[0, 0] = 500
intrinsics[1, 1] = 500

intrinsics[0, 2] = 512/2
intrinsics[1, 2] = 512/2


def setup_environment(use_graphics):
    """Initializes the DressingEnv environment and returns the robot and gripper objects."""
    env = DressingEnv(graphics=True)
    robot = env.get_robot()
    env.step()
    return env, robot

def operate_gripper(gripper, env, open_gripper=True):
    """Opens or closes the gripper and advances the simulation."""
    if open_gripper:
        gripper.GripperOpen()
    else:
        gripper.GripperClose()
    env.step()

def setup_camera(camera, gripper):
    """Configures the camera position and retrieves RGB, normal, and depth data."""
    camera.SetTransform(position=[0.45, 2.1, 0.5], rotation=[150, -90, 180])
    # camera.SetParent(gripper.id)  # Assegurar que o ID do gripper esteja correto
    camera.GetRGB(512, 512)
    camera.GetNormal(512, 512)
    camera.GetDepth(0.1, 2.0, 512, 512)


def process_images(env, camera):
    """Processes the RGB, normal, and depth images captured by the camera."""
    env.step()
    rgb = np.frombuffer(camera.data["rgb"], dtype=np.uint8)
    rgb = cv2.imdecode(rgb, cv2.IMREAD_COLOR)
    cv2.imwrite("rgb_hand.png", rgb)

    normal = np.frombuffer(camera.data["normal"], dtype=np.uint8)
    normal = cv2.imdecode(normal, cv2.IMREAD_COLOR)
    cv2.imwrite("normal_hand.png", normal)

    depth = np.frombuffer(camera.data["depth"], dtype=np.uint8)
    depth = cv2.imdecode(depth, cv2.IMREAD_GRAYSCALE)
    depth = depth.astype(np.float32) / 255.0  # Convert to depth values in meters
    np.save("depth_hand.npy", depth)

    return rgb, normal, depth



def world2camera(world_position : ArrayLike, camera : CameraAttr, intrinsics: ArrayLike) -> np.ndarray:
    P = np.zeros((3,4))
    P[:3,:3] = intrinsics

    T = np.array(camera.data["local_to_world_matrix"])
    T = np.linalg.inv(T)

    P = P @ T

    X = np.empty(4)
    X[:3] = world_position
    X[3] = 1

    x = P @ X
    x /= x[2]

    point = x[:2].astype(int)
    point[1] = 512 - point[1]

    return point

def get_normal_from_camera(camera, particles_camera):
    """Obtains the surface normal from the normal image captured by the camera."""
    normal_image = np.frombuffer(camera.data["normal"], dtype=np.uint8)
    normal_image = cv2.imdecode(normal_image, cv2.IMREAD_COLOR)
    
    # Convert the target position from 3D coordinates to 2D image coordinates
    height, width, _ = normal_image.shape
    u = int(particles_camera[0] * width) % width
    v = int(particles_camera[1] * height) % height

    # Ensure u and v are within image bounds
    u = np.clip(u, 0, width - 1)
    v = np.clip(v, 0, height - 1)

    # Get the surface normal at (u, v)
    normal = normal_image[v, u, :]
    normal = (normal / 255.0) * 2 - 1  # Convert from [0, 255] to [-1, 1]
    return normal

def move_robot_to_point(robot, target_position, grasping_normal, gripper, env):
    """Moves the robot to the grasping point with the calculated orientation and closes the gripper."""
    # Define the gripper orientation
    y_axis = grasping_normal
    x_axis = np.array([1, 0, 0])  # Arbitray vector, adjust as needed
    z_axis = np.cross(y_axis, x_axis) 
    x_axis = np.cross(z_axis, y_axis)  # Ensrure x, y, z are mutually orthogonal

    rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))
    rotation_angles = rotation_matrix_to_euler_angles(rotation_matrix)
    
    # Apply the calculated rotation after closing the gripper

    rotation_angles[2] -= 20 
    robot.IKTargetDoRotate(rotation=rotation_angles, duration=2, speed_based=False)
    robot.WaitDo()

    # Move the robot to the grasping point
    robot.IKTargetDoMove(position=target_position, duration=2, speed_based=False)
    robot.WaitDo()

    # Close the gripper upon reaching the point
    gripper.GripperClose()
    env.step()

def rotation_matrix_to_euler_angles(R):
    """Converts a rotation matrix to Euler angles."""
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0

    return np.degrees([x, y, z])


def complete_initial_task(env, robot, gripper):
    """Completes the initial task of moving the robot to the grasping point."""

    # Initial position
    robot.EnabledNativeIK(False)
    robot.SetJointPositionDirectly([173.68829345703125, -33.00761795043945, 179.9871063232422, -91.84587860107422, -39.07077407836914, -63.777915954589844, -136.49273681640625])
    env.step(50)
    robot.EnabledNativeIK(True)

    # Camera setup
    camera = env.get_camera()
    setup_camera(camera, gripper)
    rgb, normal, depth = process_images(env, camera)
    
    # Define the desired grasping position
    target_position = [1.9140679121017456, 1.7068316078186035, 0.3185024857521057]
    particles_camera = world2camera(target_position, camera, intrinsics)
    # Calculate the surface normal at the grasping point
    grasping_normal = get_normal_from_camera(camera, particles_camera)
    
    # Move the robot to the grasping point and adjust orientation
    move_robot_to_point(robot, target_position, grasping_normal, gripper, env)
    env.step()    
    
    
    

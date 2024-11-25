import numpy as np
from gymnasium import Env, spaces
from pyrcareworld.envs.dressing_env import DressingEnv
from environment.normalization import normalize_angle_cyclic, normalize_angle_with_limits
from task_ini import complete_initial_task

GRAPHICS = True

class DressEnvGAIL(Env):
    def __init__(self):
        super(DressEnvGAIL, self).__init__()
        self.env = DressingEnv(graphics=GRAPHICS)
        self.t = 0
        self.joint_positions = np.zeros(7)
        self.gripper_position = np.zeros(3)
        self.gripper_orientation = np.zeros(3)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(13,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(39,),
            dtype=np.float32
        )

        self.previous_states = [self._get_initial_observation()] * 3

    def _get_initial_observation(self):
        return np.concatenate([
            self.joint_positions,
            self.gripper_position,
            self.gripper_orientation
        ])

    def _normalize_real_data(self):
        self.joint_positions = [
            normalize_angle_with_limits(angle, JOINT_LIMITS[i + 1]) 
            for i, angle in enumerate(self.joint_positions)
        ]
        self.gripper_orientation = [
            normalize_angle_cyclic(angle) for angle in self.gripper_orientation
        ]

    def reset(self, seed=None, options=None):
        self.env.close()
        self.env = DressingEnv(graphics=GRAPHICS)
        self.t = 0

        self.kinova_id = 315893
        self.robot = self.env.GetAttr(self.kinova_id)
        self.gripper = self.env.GetAttr(3158930)

        # Completar a tarefa inicial antes do treinamento
        complete_initial_task(self.env, self.robot, self.gripper)
        
        self.robot.EnabledNativeIK(False)
        self.joint_positions = np.zeros(7)
        self.gripper_position = self.gripper.data['position']
        self.gripper_orientation = self.gripper.data['rotation']
        
        self.previous_states = [self._get_initial_observation()] * 3

        return self._get_observation(), {}

    def step(self, action):
        delta_joint_positions = action[:7]
        delta_gripper_position = action[7:10]
        delta_gripper_orientation = action[10:13]

        self.joint_positions += delta_joint_positions
        self.gripper_position += delta_gripper_position
        self.gripper_orientation += delta_gripper_orientation

        self.robot.SetJointPosition(self.joint_positions)
        self.env.step()

        self.gripper_position = self.gripper.data['position']
        self.gripper_orientation = self.gripper.data['rotation']

        self._normalize_real_data()

        observation = self._get_initial_observation()
        self.previous_states.pop(0)
        self.previous_states.append(observation)

        reward = 0.0
        done = self.t > 250  # Terminar o episódio após 300 passos
        self.t += 1
        info = {}
        return self._get_observation(), reward, done, info


    def _get_observation(self):
        obs = np.concatenate(self.previous_states)
        assert obs.shape[0] == 39, f"Expected observation shape (39,), but got {obs.shape}"
        return obs

    def render(self, mode="human"):
        pass

    def close(self):
        self.env.close()
import os
from typing import Sequence

import omni.isaac.lab.sim as sim_utils
import torch
from loguru import logger as log
from omni.isaac.lab.assets import RigidObject, RigidObjectCfg
from omni.isaac.lab.utils import configclass

from src.checkers.metrics import ProximityMetric, RobotJointVelMetric
from src.checkers.operators import AndOperator
from src.checkers.spatial import SphereProximityChecker
from src.tasks.base.base_task_env import BaseTaskEnv
from src.tasks.base.base_task_env_cfg import BaseTaskEnvCfg
from src.utils.tensor_util import tensor_to_str


@configclass
class RandomMovementEnvCfg(BaseTaskEnvCfg):
    decimation: int = 3
    episode_length_s: float = 300 * (1 / 60) * decimation  # 300 steps for random movement
    demo_file_path: str = "data_isaaclab/source_data/demo/random_movement_demo.pkl"


class RandomMovementEnv(BaseTaskEnv):
    cfg: RandomMovementEnvCfg

    def add_object(self):
        # no object needed for random movement task
        pass

    def init_buffer(self):
        super().init_buffer()
        self.random_init = torch.zeros((self.num_envs, 3), device=self.device)
        self.random_target = torch.zeros((self.num_envs, 3), device=self.device)
        self.random_positions = torch.zeros((self.num_envs, self.max_episode_length, 3), device=self.device)
        # Assume demo_data is loaded from file in main and assigned to self.demo_data
        # e.g., self.demo_data = torch.load(self.cfg.demo_file_path)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        self.robot_state = self.robot.data.body_state_w[:, self.robot.data.body_names.index("panda_hand")]
        is_success = torch.zeros_like(time_out)
        # Optionally add custom success metrics for random movement here
        return is_success, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        super()._reset_idx(env_ids)
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        demo_idxs = self.demo_idxs[env_ids]
        # Replay the demo trajectory loaded in main
        # Assume self.demo_data["trajectory"] shape: (num_demos, max_episode_length, 3)
        self.random_positions[env_ids, :, :] = self.demo_data["trajectory"][demo_idxs]
        self.random_init[env_ids] = self.random_positions[env_ids, 0, :]
        self.random_target[env_ids] = self.random_positions[env_ids, -1, :]
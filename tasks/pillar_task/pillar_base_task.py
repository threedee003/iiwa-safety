from workspace.iiwa_pillar_env import PillarEnv
from configs.task_configs import *

from isaacgym import gymapi




class PillarBaseTask(PillarEnv):
      def __init__(self):
            super().__init__()
            self.task_cfg = None
      
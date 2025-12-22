from configs.asset_config import AssetConfig
from workspace.iiwa_testbed import iiwaTestBed

from isaacgym import gymapi


class PillarEnv(iiwaTestBed):
      def __init__(self):
            super().__init__()
            self.asset_cfg = AssetConfig()
            self.pillar_poses = []
            self.pillar_assets = []
            self.pillar_handles = []
            self.create_pillar()
            for i in range(len(self.pillar_assets)):
                  pillar_handle = self.gym.create_actor(self.env, self.pillar_assets[i], self.pillar_poses[i], f"pillar_{i+1}", 0, 0)
                  self.pillar_handles.append(pillar_handle)

            self.gym.prepare_sim(self.sim)



      def create_pillar(self):
            pillar_dim = self.asset_cfg.pillar.pillar_dim
            pillar_pos = self.asset_cfg.pillar.pillar_position
            pillar_orientation = self.asset_cfg.pillar.pillar_orientation
            asset_options = gymapi.AssetOptions()
            asset_options.fix_base_link = True

            pillar_pose = gymapi.Transform()
            pillar_pose.p = gymapi.Vec3(pillar_pos[0], pillar_pos[1], pillar_pos[2])
            pillar_ = self.gym.create_box(self.sim, pillar_dim[0], pillar_dim[1], pillar_dim[2], asset_options)
            self.pillar_poses.append(pillar_pose)
            self.pillar_assets.append(pillar_)

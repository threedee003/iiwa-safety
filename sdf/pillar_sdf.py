from configs.asset_config import AssetConfig

import torch
import numpy as np





class PillarSDF:
      def __init__(self):
            self.asset_cfg = AssetConfig()
            self.pillar_pos = self.asset_cfg.pillar.pillar_position
            self.pillar_quat = self.asset_cfg.pillar.pillar_orientation
            self.pillar_dim = self.asset_cfg.pillar.pillar_dim


      def sdf_3d_outside(self, position):
            x, y, z = position[0], position[1], position[2]
            dx = max(x - self.pillar_dim[0], 0)
            dy = max(y - self.pillar_dim[1], 0)
            dz = max(y - self.pillar_dim[2], 0)
            dist = np.sqrt(dx**2 + dy**2 + dz**2)
            return dist
            
      def sdf_3d_inside(self, position):





      # def sdf_2d(self, position):
      #       x, y = position[0], position[1] # we consider the position is in 
      #       d = torch.sqrt(max(x - self.pillar_dim[0], 0)**2 + max(y - self.pillar_dim[1], 0)**2)
      #       return d
      



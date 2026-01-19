from configs.asset_config import AssetConfig

import torch
import numpy as np





class PillarSDF:
      def __init__(self):
            self.asset_cfg = AssetConfig()
            self.pillar_pos = self.asset_cfg.pillar.pillar_position
            self.pillar_quat = self.asset_cfg.pillar.pillar_orientation
            self.pillar_dim = self.asset_cfg.pillar.pillar_dim

            self.voxel_size = self.asset_cfg.pillar.sdf_config.voxel_size
            self.x_min = self.asset_cfg.pillar.sdf_config.x_min
            self.x_max = self.asset_cfg.pillar.sdf_config.x_max
            self.z_min = self.asset_cfg.pillar.sdf_config.z_min
            self.z_max = self.asset_cfg.pillar.sdf_config.z_max
            self.y_min = self.asset_cfg.pillar.sdf_config.y_min
            self.y_max = self.asset_cfg.pillar.sdf_config.y_max
            self.Nx = self.asset_cfg.pillar.sdf_config.Nx
            self.Ny = self.asset_cfg.pillar.sdf_config.Ny
            self.Nz = self.asset_cfg.pillar.sdf_config.Nz
            self.device = self.asset_cfg.pillar.sdf_config.device
            self.sdf = torch.zeros(size=(self.Nx, self.Ny, self.Nz), device=self.device)


            xs = (torch.arange(self.Nx, device = self.device).float() + 0.5) * self.voxel_size + self.x_min
            ys = (torch.arange(self.Ny, device = self.device).float() + 0.5) * self.voxel_size + self.y_min
            zs = (torch.arange(self.Nz, device = self.device).float() + 0.5) * self.voxel_size + self.z_min
            gx, gy, gz = torch.meshgrid(xs, ys, zs, indexing = 'ij')
            self.voxel_cordinates = torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim = 1)

            

      def sdf_3d_outside(self, position):
            x, y, z = position[0], position[1], position[2]
            dx = max(np.abs(x) - self.pillar_dim[0], 0)
            dy = max(np.abs(y) - self.pillar_dim[1], 0)
            dz = max(np.abs(z) - self.pillar_dim[2], 0)
            dist = np.sqrt(dx**2 + dy**2 + dz**2)
            return dist
            
      def sdf_3d_inside(self, position):
            x, y, z = position[0], position[1], position[2]
            dx = min(max(np.abs(x) - self.pillar_dim[0]), 0)
            dy = min(max(np.abs(y) - self.pillar_dim[1]), 0)
            dz = min(max(np.abs(z) - self.pillar_dim[2]), 0)
            dist = np.sqrt(dx**2 + dy**2 + dz**2)
            return dist
      





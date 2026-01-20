from configs.asset_config import AssetConfig

import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm



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
            self.slice_dir = self.asset_cfg.pillar.sdf_config.slice_dir


            xs = (torch.arange(self.Nx, device = self.device).float() + 0.5) * self.voxel_size + self.x_min
            ys = (torch.arange(self.Ny, device = self.device).float() + 0.5) * self.voxel_size + self.y_min
            zs = (torch.arange(self.Nz, device = self.device).float() + 0.5) * self.voxel_size + self.z_min
            gx, gy, gz = torch.meshgrid(xs, ys, zs, indexing = 'ij')
            self.voxel_cordinates = torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim = 1)
            # print(xs)
            self.compute_sdf()
            self.save_esdf_slices(axes='z')
            print(self.sdf[self.Nx // 2, self.Ny // 2, self.Nz // 2])
            



      def sdf_3d(self, positions):
            """
            positions: (N, 3) world-frame positions
            returns:   (N,) signed distance
            """

            # shift to pillar local frame
            center = torch.tensor(
                  self.pillar_pos, device=self.device
            )
            p_local = positions - center

            # box SDF in local frame
            q = torch.abs(p_local) - 0.5 * torch.tensor(
                  self.pillar_dim, device=self.device
            )

            outside = torch.linalg.norm(torch.clamp(q, min=0.0), dim=1)
            inside = torch.clamp(torch.max(q, dim=1).values, max=0.0)

            return outside + inside

      
      def compute_sdf(self):
            sdf_flat = self.sdf_3d(self.voxel_cordinates)
            self.sdf = sdf_flat.view(self.Nx, self.Ny, self.Nz)





      def save_esdf_slices(self, axes='z'):

            os.makedirs(self.slice_dir, exist_ok=True)

            if axes == "z":
                  n = self.Nz
                  s1, s2 = 'X', 'Y'
            else:
                  n = self.Nx
                  s1, s2 = 'Y', 'Z'

            print("Starting SDF saving")

            # global min/max for consistent coloring
            sdf_min = self.sdf.min().item()
            sdf_max = self.sdf.max().item()

            # center colormap at zero
            norm = TwoSlopeNorm(vmin=sdf_min, vcenter=0.0, vmax=sdf_max)

            for i in range(n):
                  if axes == 'z':
                        slice = self.sdf[:, :, i].cpu().numpy()
                  else:
                        slice = self.sdf[i, :, :].cpu().numpy()

                  plt.figure(figsize=(8, 7))
                  im = plt.imshow(
                        slice.T,
                        origin="lower",
                        cmap="coolwarm",   # negative = blue, positive = red
                        norm=norm
                  )

                  cbar = plt.colorbar(im)
                  cbar.set_label("Signed distance (m)")
                  cbar.ax.axhline(0, color='k', linewidth=1)  # mark zero level

                  plt.title(f"ESDF {s1 + s2} slice index = {i}")
                  plt.xlabel(f"{s1} index")
                  plt.ylabel(f"{s2} index")

                  plt.savefig(
                        os.path.join(self.slice_dir, f"sdf_{s1+s2}_z{i:03d}.png"),
                        dpi=300,
                        bbox_inches="tight"
                  )
                  plt.close()

            print(f"{n} ESDF slices saved")




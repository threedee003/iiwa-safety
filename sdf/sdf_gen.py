from configs.sdf_configs import SDFConfig
from configs.camera_configs import CameraConfig

import numpy as np
import os
import torch
from typing import Optional
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt



class SDFVolume:
      def __init__(self, 
                   K_cam: Optional[list] = None,
                   resolution: Optional[list] = [640, 480]
                  ):
            self.sdf_cfg = SDFConfig()
            self.cam_cfg = CameraConfig()
            self.resolution = resolution
            self.K = K_cam
            self.voxel_size = self.sdf_cfg.shelf_env.voxel_size
            self.x_min = self.sdf_cfg.shelf_env.x_min
            self.x_max = self.sdf_cfg.shelf_env.x_max
            self.y_min = self.sdf_cfg.shelf_env.y_min
            self.y_max = self.sdf_cfg.shelf_env.y_max
            self.z_min = self.sdf_cfg.shelf_env.z_min
            self.z_max = self.sdf_cfg.shelf_env.z_max
            
            self.Nx = self.sdf_cfg.shelf_env.Nx
            self.Ny = self.sdf_cfg.shelf_env.Ny
            self.Nz = self.sdf_cfg.shelf_env.Nz
            self.num_voxels = self.Nx * self.Ny * self.Nz
            self.device = self.sdf_cfg.shelf_env.device

            self.tsdf = torch.ones((self.Nx, self.Ny, self.Nz), device = self.device)
            self.weight = torch.zeros_like(self.tsdf).to(self.device)
            self.esdf = torch.ones_like(self.tsdf).to(self.device)

            xs = (torch.arange(self.Nx, device = self.device).float() + 0.5) * self.voxel_size + self.x_min
            ys = (torch.arange(self.Ny, device = self.device).float() + 0.5) * self.voxel_size + self.y_min
            zs = (torch.arange(self.Nz, device = self.device).float() + 0.5) * self.voxel_size + self.z_min

            gx, gy, gz = torch.meshgrid(xs, ys, zs, indexing = 'ij')
            self.voxel_cordinates = torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim = 1)
            self.frame = 0
            self.debug()
            voxel_cord = self.voxel_cordinates.cpu().numpy()
            print(voxel_cord.shape)
            np.savetxt("./voxel_cord.txt", voxel_cord)


      def debug(self):
            i, j, k = 0, 0, 0
            idx = i * (self.Ny * self.Nz) + j * self.Nz + k
            print(f"debug : {self.voxel_cordinates[idx]}") 






      def depth2world(self, depth, seg, T_wc):
            if not isinstance(depth, torch.Tensor):
                  depth = torch.from_numpy(depth).to(self.device)
            if not isinstance(seg, torch.Tensor):
                  seg = torch.from_numpy(seg).to(self.device)
            if not isinstance(T_wc, torch.Tensor):
                  T_wc = torch.from_numpy(T_wc).to(self.device)



            fx, fy, cx, cy = self.K[0], self.K[1], self.K[2], self.K[3]
            H, W = self.resolution[1], self.resolution[0]

            v, u = torch.meshgrid(
                  torch.arange(H, device = self.device),
                  torch.arange(W, device = self.device),
                  indexing = 'ij'
            )
            z = depth
            valid = (z > 0.05) & (seg != 1)
            if valid.sum() == 0:
                  return None
            
            u = u[valid].float()
            v = v[valid].float()
            z = z[valid]

            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            
            pc_cam = torch.stack(
                  [x, y, z, torch.ones_like(z)],
                  dim = 1
            )

            pc_world = (torch.linalg.inv(T_wc) @ pc_cam.T).T[:,:3]
            return pc_world
      
            
      def world2grid(self, points):
            if not isinstance(points, torch.Tensor):
                  points = torch.from_numpy(points)
            g = (points - torch.tensor(
                  [self.x_min, self.y_min, self.z_min],
                  device=self.device
            )) / self.voxel_size


            idx = torch.floor(g).long()
            idx[:, 0].clamp_(0, self.Nx - 1)
            idx[:, 1].clamp_(0, self.Ny - 1)
            idx[:, 2].clamp_(0, self.Nz - 1)
            return idx

      # def integrate(self, depth, seg, T_wc):
      #       pc_world = self.depth2world(depth, seg, T_wc)
      #       if pc_world is None:
      #             return
            
      #       idx = self.world2grid(pc_world)
      #       i, j, k = idx[:, 0], idx[:, 1], idx[:, 2]
      #       centers = torch.stack([
      #             self.x_min + (i.float() + 0.5) * self.voxel_size,
      #             self.y_min + (j.float() + 0.5) * self.voxel_size,
      #             self.z_min + (k.float() + 0.5) * self.voxel_size
      #       ], dim = 1)
      #       sdf = torch.norm(pc_world - centers, dim = 1)
      #       mu = 3. * self.voxel_size
      #       phi = torch.clamp(sdf / mu, -1., 1.)
      #       w_old = self.weight[i, j, k]
      #       t_old = self.tsdf[i, j, k]

      #       w_new = torch.clamp(w_old + 1., max = 50.)
      #       t_new = (w_old * t_old + phi) / w_new

      #       self.tsdf[i, j, k] = t_new
      #       self.weight[i, j, k] = w_new

      #       self.frame += 1
      #       if self.frame % 10 == 0:
      #             self.update_esdf()

      def integrate(self, depth, seg, T_wc):
            if not isinstance(depth, torch.Tensor):
                  depth = torch.from_numpy(depth).to(self.device)
            if not isinstance(seg, torch.Tensor):
                  seg = torch.from_numpy(seg).to(self.device)
            if not isinstance(T_wc, torch.Tensor):
                  T_wc = torch.from_numpy(T_wc).to(self.device)

            # world <- camera
            T_cw = torch.linalg.inv(T_wc)

            fx, fy, cx, cy = self.K
            H, W = depth.shape

            v, u = torch.meshgrid(
                  torch.arange(H, device=self.device),
                  torch.arange(W, device=self.device),
                  indexing="ij"
            )

            z = depth
            valid = (z > 0.05) & (seg == 0)
            if valid.sum() == 0:
                  return

            u = u[valid].float()
            v = v[valid].float()
            z = z[valid]

 
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy

            pc_cam = torch.stack(
                  [x, y, z, torch.ones_like(z)],
                  dim=1
            )  

        
            pc_world = (T_cw @ pc_cam.T).T[:, :3]

           
            idx = self.world2grid(pc_world)
            i, j, k = idx[:, 0], idx[:, 1], idx[:, 2]

        
            centers = torch.stack([
                  self.x_min + (i.float() + 0.5) * self.voxel_size,
                  self.y_min + (j.float() + 0.5) * self.voxel_size,
                  self.z_min + (k.float() + 0.5) * self.voxel_size,
            ], dim=1)

        
            cam_origin = T_cw[:3, 3]

            ray_dir = pc_world - cam_origin
            ray_dir = ray_dir / torch.norm(ray_dir, dim=1, keepdim=True)

            voxel_vec = centers - cam_origin
            proj = (voxel_vec * ray_dir).sum(dim=1)

            sdf = torch.norm(voxel_vec, dim=1) - proj

            mu = 3.0 * self.voxel_size
            phi = torch.clamp(sdf / mu, -1.0, 1.0)


            w_old = self.weight[i, j, k]
            t_old = self.tsdf[i, j, k]

            w_new = torch.clamp(w_old + 1.0, max=50.0)
            t_new = (w_old * t_old + phi) / w_new

            self.tsdf[i, j, k] = t_new
            self.weight[i, j, k] = w_new

            self.frame += 1
       
            self.update_esdf()


            print("TSDF min:", self.tsdf.min().item())
            print("TSDF max:", self.tsdf.max().item())
            print("Updated voxels:", (self.weight > 0).sum().item())
      









      def update_esdf(self):
            tsdf_cpu = self.tsdf.detach().cpu().numpy()
            occ = tsdf_cpu < 0

            dist_out = distance_transform_edt(~occ)
            dist_in = distance_transform_edt(occ)
            esdf_cpu = (dist_out - dist_in) * self.voxel_size
            self.esdf = torch.from_numpy(esdf_cpu).to(self.device)





      def query(self, points):
            if not isinstance(points, torch.Tensor):
                  points = torch.from_numpy(points)
            p = points.to(self.device)
            idx =  self.world2grid(p)
            i, j, k = idx[:, 0], idx[:, 1], idx[:, 2]; d = self.esdf[i, j, k]
            
            im = (i - 1).clamp(0, self.Nx - 1); ip = (i + 1).clamp(0, self.Nx - 1)
            jm = (j - 1).clamp(0, self.Ny - 1); jp = (j + 1).clamp(0, self.Ny - 1)
            km = (k - 1).clmap(0, self.Nz - 1); kp = (k + 1).clamp(0, self.Nz - 1)


            gx = (self.esdf[ip, j, k] - self.esdf[im, j, k]) / (2 * self.voxel_size)
            gy = (self.esdf[i, jp, k] - self.esdf[i, jm, k]) / (2 * self.voxel_size)
            gz = (self.esdf[i, j, kp] - self.esdf[i, j, km]) / (2 * self.voxel_size)

            grad = torch.stack([gx, gy, gz], dim = 1)
            return d, grad
      


      def save_esdf_slices(self, axes = 'z'):

            os.makedirs(self.sdf_cfg.shelf_env.slice_dir, exist_ok = True)
            if axes == "z":
                  n = self.Nz
                  s1, s2 = 'X', 'Y'
            else:
                  n = self.Nx
                  s1, s2 = 'Y', 'Z'
            print("Starting SDF saving")
            for i in range(n):
                  if axes == 'z':
                        slice = self.esdf[:,:,i].cpu().numpy()
                  else:
                        slice = self.esdf[i, :, :].cpu().numpy()
                  plt.figure(figsize=(8,7))
                  plt.imshow(slice.T, origin="lower", cmap="coolwarm")
                  plt.colorbar(label = "distance (m)")
                  plt.title(f"ESDF {s1 + s2} slice z = {i}")
                  plt.xlabel(f"{s1} index")
                  plt.ylabel(f"{s2} index")
                  plt.savefig(
                        os.path.join(self.sdf_cfg.shelf_env.slice_dir, f"esdf_{s1+s2}_z{i:03d}.png"),
                        dpi = 300,
                        bbox_inches = "tight"
                  )
                  plt.close()
            print(f"{n} ESDF slices saved")






      def show_xy_slice(self, z_idx, title="ESDF XY slice"):
            slice_xy = self.esdf[:, :, z_idx].cpu().numpy()
            plt.imshow(slice_xy.T, origin="lower", cmap="coolwarm")
            plt.colorbar(label="distance (m)")
            plt.title(title)
            plt.xlabel("X index")
            plt.ylabel("Y index")
            plt.show()

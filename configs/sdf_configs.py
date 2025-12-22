
class CameraConfigs:
      class right_cam:
            position = [-0.2, -1, 1.8]
            lookat = [0.8, 0.2, 1.6]
            resolution = [640, 480]
            fov = 75.
            


      class left_cam:
            position = [-0.2, 1, 1.8]
            lookat = [0.8, 0.2, 1.6]
            resolution = [640, 480]
            fov = 75.

      class depth_process:
            clip_dist = 3.

      


class SDFConfig:
      class shelf_env:
            voxel_size = 0.02
            x_min = -1.3 
            x_max =  1.3
            y_min = -1.5
            y_max = 1.5
            z_min = 0.
            z_max = 2.2
            Nx = int((x_max - x_min) / voxel_size)
            Ny = int((y_max - y_min) / voxel_size)
            Nz = int((z_max - z_min) / voxel_size)
            device = 'cuda'
            slice_dir = "/home/bikram/Documents/isaacgym/iiwa_safety/sdf_slices/shelf_env"
from isaacgym import gymapi
import numpy as np


'''
Quaternions are in format x y z w
'''

class AssetConfig:
      class shelf:
            # width, length, thickness
            shelf_plank_dim = [0.4, 1.5, 0.04]
            # please keep the num planks upto 3 or 4
            # depending on the plank gap for reachability
            num_planks = 4
            plank_gap = 0.3
            shelf_position = [0.8, -0.7, 1.]
            shelf_orientation = [0., 0., 0., 1.]



      class pillar:
            # [cs_width, cs_length, height] NOTE: cs == cross-section
            pillar_dim = [0.5, 0.5, 1.5] 
            pillar_position = [0.8, 0, 0.8]
            pillar_orientation = [0., 0., 0., 1.] 

            class sdf_config:
                  voxel_size = 0.05
                  x_min = -1.6 
                  x_max =  1.6
                  y_min = -1.5
                  y_max = 1.5
                  z_min = 0
                  z_max = 2.2
                  Nx = int((x_max - x_min) / voxel_size)
                  Ny = int((y_max - y_min) / voxel_size)
                  Nz = int((z_max - z_min) / voxel_size)
                  device = "cuda"
                  slice_dir = "/home/bikram/Documents/isaacgym/iiwa_safety/sdf_slices/pillar_env"
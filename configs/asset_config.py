from isaacgym import gymapi
import numpy as np




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
            pillar_dim = [0.5, 0.5, 1.6]
            pillar_position = [0.8, 0, 0.8]
            pillar_orientation = [0., 0., 0., 1.]



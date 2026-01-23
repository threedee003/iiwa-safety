
from dynamics.base_dynamics import DynamicsModel

import torch




class IKDynamicsModel(DynamicsModel):
      def __init__(self):
            super().__init__()
            pass



      def _get_joint_position(self, j_dot: torch.Tensor, j_curr: torch.Tensor, del_t: float) -> torch.Tensor:
            j_new = j_curr + del_t * j_dot
            return j_new


      def _get_link_positions(self):
            raise NotImplementedError("This function will be used from GymAPI")
      

      def _get_eef_positions(self, joint_positions):

from tasks.pillar_task.pillar_base_task import PillarBaseTask
from dynamics.ik_dynamics import IKDynamicsModel
from mppi.algorunner import AlgoRunner
import torch

class PillarReachAvoid(PillarBaseTask):
      def __init__(self):
            self.start_pos = None
            self.goal_pos = None
            self.current_eef_pos = None
            self.dynamicsM = IKDynamicsModel()
            self.algo = AlgoRunner(algo_name="MPPI").algo






      def reach_cost(self, current_eef_pos):
            """
            Docstring for reach_cost
      
            :param current_eef_pos: Imagined current eef pos from the WM.
            :param current_joint_position: Imagined joint position from the WM. 
            """
            if not isinstance(current_eef_pos, torch.Tensor):
                  current_eef_pos = torch.from_numpy(current_eef_pos)
            cost = torch.abs(self.goal_pos - current_eef_pos)
            return cost
      



      def avoid_cost(self, links):
            
            link_pos = self.dynamicsM._get_link_positions()

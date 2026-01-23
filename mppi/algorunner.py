from mppi.algorithm import MPPI, SMPPI
import torch


class AlgoRunner:
      def __init__(self, algo_name="MPPI"):
            if algo_name == "MPPI":
                  self.algo = MPPI()
            elif algo_name == "SMPPI":
                  self.algo = SMPPI()


      def get_algo(self):
            return self.algo
            
                  


      
import torch
from dynamics.base_dynamics import DynamicsModel

class MPPI:
      """
      MPPI == Model Predictive Path Integral Control

      """
      def __init__(self,
            dynamics_model: DynamicsModel,
            device: torch.device,
            noise_sigma: float,
            noise_mu: float,
            num_samples: int,
            horizon: int,
            terminal_state_cost: bool,
            lambda_ : float,
            u_min: list,
            u_max: list,
            u_scale: float,
            u_per_command: int,
            step_dependent_dynamics: bool,
            rollout_samples: int
      ):    
            self.F = dynamics_model
            self.device = device
            self.noise_sigma = noise_sigma
            self.noise_mu = noise_mu
            self.K = num_samples
            self.T = horizon
            self.terminal_state_cost = terminal_state_cost
            self.labmda_ = lambda_
            self.M = rollout_samples
            self.u_min = u_min
            self.u_max = u_max
            self.u_scale = u_scale
            self.u_per_command = u_per_command
            self.step_dependent_dynamics = step_dependent_dynamics


      def _dynamics(self, state: torch.Tensor, u: torch.Tensor):
            next_state = self.F(state, u)
            return next_state
      



class SMPPI:
      """
      MPPI == Model Predictive Path Integral Control

      """
      def __init__(self,
            dynamics_model: DynamicsModel,
            device: torch.device,
            noise_sigma: float,
            noise_mu: float,
            num_samples: int,
            horizon: int,
            terminal_state_cost: bool,
            lambda_ : float,
            u_min: list,
            u_max: list,
            u_scale: float,
            u_per_command: int,
            step_dependent_dynamics: bool,
            rollout_samples: int
      ):    
            self.F = dynamics_model
            self.device = device
            self.noise_sigma = noise_sigma
            self.noise_mu = noise_mu
            self.K = num_samples
            self.T = horizon
            self.terminal_state_cost = terminal_state_cost
            self.labmda_ = lambda_
            self.M = rollout_samples
            self.u_min = u_min
            self.u_max = u_max
            self.u_scale = u_scale
            self.u_per_command = u_per_command
            self.step_dependent_dynamics = step_dependent_dynamics
      

      def _dynamics(self, state: torch.Tensor, u: torch.Tensor):
            next_state = self.F(state, u)
            return next_state
      



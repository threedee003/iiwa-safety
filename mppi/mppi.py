import torch


class MPPI:
      """
      MPPI == Model Predictive Path Integral Control

      """
      def __init__(self,
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
            self.device = device
            self.

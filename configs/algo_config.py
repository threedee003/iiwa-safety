

class MPPICfg:
      class algo_params:
            noise_sigma = 0.03
            noise_mu = 1
            num_samples = 30
            horizon = 10
            terminal_state_cost = False
            lambda_ = 0.02
            u_min = 7*[-1.]
            u_max = 7*[1.]
            u_scale = 1.
            u_per_command = 1
            step_dependent_dynamics = False
            rollout_samples = 2

      class cost_params:
            pass


      class dynamics_param:
            pass
      


      class device_params:
            device = 'cuda'

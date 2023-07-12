def process_env(args):
    ### Environment initialization ###
    if args.environment == 'ball_in_cup':
        obs_dim = 6
        act_dim = 3
        dim_map = 3
        max_step = 300
        obs_min = np.array([-0.4]*6)
        obs_max = np.array([0.4]*6)
    elif args.environment == 'redundant_arm':
        dim_map = 2
    elif args.environment == 'redundant_arm_no_walls':
        dim_map = 2
    elif args.environment == 'redundant_arm_no_walls_no_collision':
        dim_map = 2
    elif args.environment == 'redundant_arm_no_walls_limited_angles':
        dim_map = 2
    elif args.environment == 'fastsim_maze_laser':
        dim_map = 2
    elif args.environment == 'empty_maze_laser':
        dim_map = 2
    elif args.environment == 'fastsim_maze':
        dim_map = 2
    elif args.environment == 'empty_maze':
        dim_map = 2
    elif args.environment == 'fastsim_maze_traps':
        dim_map = 2
    elif args.environment == 'half_cheetah':
        dim_map = 1
    elif args.environment == 'walker2d':
        dim_map = 1
    elif args.environment == 'hexapod_omni':
        dim_map = 2
    elif args.environment == 'cartpole':
        obs_dim = 4
        act_dim = 1
        dim_map = 2
        max_step = 200
        obs_min = np.array(np.array([-2.5, -8, -8, 15]))
        obs_max = np.array([2.5, 8, 8, 15])
        
    elif args.environment == 'pusher':
        obs_dim = 20
        act_dim = 7
        dim_map = 6
        max_step = 150
        obs_min = np.array([-2., -1.5, -1.5,
                            -2, -1.5, -1,
                            -1.5, -1.5,
                            -1.5, -8, -5,
                            -7, -5, -7, -1,
                            -1.5, -0.5, -1,
                            -1, -1])
        obs_max = np.array([2., 1.5, 1.5, 2,
                            1.5, 1, 1.5, 1.5,
                            1.5, 8, 5, 7, 5,
                            7, 1, 1.5, 0.5,
                            1, 1, 1])
    elif args.environment == 'reacher':
        obs_dim = 17
        act_dim = 7
        dim_map = 3
        max_step = 150
        obs_min = np.array([-2., -1.5, -1.5,
                            -2, -1.5, -1,
                            -1.5, -1.5,
                            -1.5, -8, -5,
                            -7, -5, -7, -1,
                            -1.5, -0.5])
        obs_max = np.array([2., 1.5, 1.5, 2,
                            1.5, 1, 1.5, 1.5,
                            1.5, 8, 5, 7, 5,
                            7, 1, 1.5, 0.5])
    else:
        raise ValueError("{} is not a defined environment".format(args.environment))
    
    return obs_dim, act_dim, dim_map, max_step, obs_min, obs_max

def main(args):
    # get env info
    obs_dim, act_dim, dim_map, max_step, obs_min, obs_max = process_env(args)

    ## Replay the selected trajectories..

    # First create the env instance
    ctrl_args = DotMap(**{key: val for (key, val) in []})
    cfg = create_config(args.environment, "MPC", ctrl_args, [], '')
    cfg.pprint()
    
    gym_env = cfg.ctrl_cfg.env
    obs = gym_env.reset()

    controller_params = \
    {
        'controller_input_dim': obs_dim,
        'controller_output_dim': act_dim,
        'n_hidden_layers': 2,
        'n_neurons_per_hidden': 10,
        'time_open_loop': 0,
        'norm_input': 1,
        'pred_mode': 'single',
    }

    params = \
    {
        'controller_params': controller_params,
    }

    obs_trajs = []
    ac_trajs = []

    from model_init_study.initializers.colored_noise_motion \
            import ColoredNoiseMotion

    init = 'colored'

    if init == 'colored':
        noise_beta = 2
        params = \
        {
            'obs_dim': obs_dim,
            'action_dim': act_dim,

            'n_init_episodes': 10,
            'n_test_episodes': 0,
            
            'action_min': gym_env.action_space.low[0],
            'action_max': gym_env.action_space.high[0],
            'action_init': 0,

            ## Random walks parameters
            'step_size': 0.1,
            'noise_beta': noise_beta,

            'env': gym_env,
            'env_name': args.environment,
            'env_max_h': max_step,
        }
        initializer = ColoredNoiseMotion(params)
        ## Execute the initializer policies on the environment
        all_trajs = initializer.run()
        ## reshape the obtained trajectories
        examples = np.empty((10, max_step+1, obs_dim))
        examples[:] = np.nan
        traj_cpt = 0
        for traj in all_trajs:
            for i in range(max_step+1):
                examples[traj_cpt,i, :] = traj[i][1]
            traj_cpt += 1
        ac_trajs = np.array(initializer.actions)
    else:
        for i in range(args.n_reps):
            obs = gym_env.reset()
            obs_trajs.append([])
            ac_trajs.append([])
            # for j in range(2):
            for t in range(max_step):
                a = gym_env.action_space.sample()
                # a = np.clip(a, -1, 1)
                obs_trajs[-1].append(obs)
                ac_trajs[-1].append(a)
                obs, r, d, info = gym_env.step(a)

            obs_trajs[-1].append(obs)

        examples = np.array(obs_trajs)
        ac_trajs = np.array(ac_trajs)

    fname = '{}_example_trajectories.npz'.format(args.environment)

    print(examples.shape)
    print(ac_trajs.shape)
    np.savez(fname,
             examples=examples,
             ac_trajs=ac_trajs)

if __name__ == '__main__':
    #----------Utils imports--------#
    import argparse
    import os

    #----------PETS env imports--------#
    from dotmap import DotMap
    from dmbrl.config import create_config

    #----------Data manipulation imports--------#
    import pandas as pd
    import numpy as np
    #----------Clustering imports--------#
    from sklearn.cluster import KMeans

    #----------Controller imports--------#
    from model_init_study.controller.nn_controller \
        import NeuralNetworkController
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--environment', '-e', type=str, required=True,
                        help='Environment name')

    parser.add_argument('--n-reps', type=int, default=10,
                        help='Number of repetitions of ns exps')
    
    args = parser.parse_args()

    main(args)

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import argparse
import pprint

from dotmap import DotMap

from dmbrl.misc.MBExp import MBExperiment
from dmbrl.controllers.MPC import MPC
from dmbrl.config import create_config


def main(env, ctrl_type, ctrl_args, overrides, logdir, init_method):
    import tensorflow as tf
    ## 1.9
    with tf.Session() as sess:
        devices = sess.list_devices()

    ## 2.1
    # devices = tf.config.list_physical_devices('GPU')
    
    print()
    print("LIST OF DEVICES: ", devices)
    print()

    ctrl_args = DotMap(**{key: val for (key, val) in ctrl_args})
    cfg = create_config(env, ctrl_type, ctrl_args, overrides, logdir)
    cfg.pprint()

    kwargs = {}
    if init_method is not None:
        ## Instantiate Initializer with params
        from model_init_study.initializers.random_policy_initializer \
            import RandomPolicyInitializer
        from model_init_study.initializers.random_actions_initializer \
            import RandomActionsInitializer
        from model_init_study.initializers.random_actions_random_policies_hybrid_initializer \
            import RARPHybridInitializer
        from model_init_study.initializers.brownian_motion \
            import BrownianMotion
        from model_init_study.initializers.colored_noise_motion \
            import ColoredNoiseMotion

        noise_beta = 2
        if args.init_method == 'random-policies':
            # Initializer = RandomPolicyInitializer
            raise NotImplementedError('Random policies not yet implemented for pets')
        elif args.init_method == 'random-actions':
            Initializer = RandomActionsInitializer
        elif args.init_method == 'rarph':
            Initializer = RARPHybridInitializer
        elif args.init_method == 'brownian-motion':
            Initializer = BrownianMotion
        elif args.init_method == 'colored-noise-beta-0':
            Initializer = ColoredNoiseMotion
            noise_beta = 0
        elif args.init_method == 'colored-noise-beta-1':
            Initializer = ColoredNoiseMotion
            noise_beta = 1
        elif args.init_method == 'colored-noise-beta-2':
            Initializer = ColoredNoiseMotion
            noise_beta = 2
        else:
            raise Exception("Warning ",args.init_method, "isn't a valid initializer")

        obs_dim = cfg.ctrl_cfg.env.obs_dim
        act_dim = cfg.ctrl_cfg.env.action_space.shape[0]
        controller_params = \
        {
            'controller_input_dim': obs_dim,
            'controller_output_dim': act_dim,
            'n_hidden_layers': 2,
            'n_neurons_per_hidden': 10
        }
        params = \
        {
            'obs_dim': obs_dim,
            'action_dim': act_dim,

            'n_init_episodes': 10,
            'n_test_episodes': 2,

            'inc_rew': True,
            # 'controller_type': NeuralNetworkController,
            # 'controller_params': controller_params,

            'action_min': cfg.exp_cfg.sim_cfg.env.action_space.low[0],
            'action_max': cfg.exp_cfg.sim_cfg.env.action_space.high[0],
            'action_init': 0,

            ## Random walks parameters
            'step_size': 0.1,
            'noise_beta': noise_beta,

            'action_lasting_steps': 5,

            'policy_param_init_min': -5,
            'policy_param_init_max': 5,

            'env': cfg.ctrl_cfg.env,
            'env_max_h': cfg.exp_cfg.sim_cfg.task_hor,
        }

        ## Instanciate the initializer
        initializer = Initializer(params)

        kwargs['initializer'] = initializer
        
    if ctrl_type == "MPC":
        cfg.exp_cfg.exp_cfg.policy = MPC(cfg.ctrl_cfg)
    
    exp = MBExperiment(cfg.exp_cfg)

    os.makedirs(exp.logdir)
    with open(os.path.join(exp.logdir, "config.txt"), "w") as f:
        f.write(pprint.pformat(cfg.toDict()))

    exp.run_experiment(**kwargs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-env', type=str, required=True,
                        help='Environment name: select from [cartpole, reacher, pusher, halfcheetah]')
    parser.add_argument('-ca', '--ctrl_arg', action='append', nargs=2, default=[],
                        help='Controller arguments, see https://github.com/kchua/handful-of-trials#controller-arguments')
    parser.add_argument('-o', '--override', action='append', nargs=2, default=[],
                        help='Override default parameters, see https://github.com/kchua/handful-of-trials#overrides')
    parser.add_argument('-logdir', type=str, default='log',
                        help='Directory to which results will be logged (default: ./log)')
    parser.add_argument('--init-method', type=str, default=None,
                        help='which initial data gathering method to use')
    
    args = parser.parse_args()

    main(args.env, "MPC", args.ctrl_arg, args.override, args.logdir, args.init_method)

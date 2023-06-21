from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

## NEED TO WRAP policy around a dynamics model obj
class WrappedPETSDynamicsModel():
    def __init__(self, policy):
        self.policy = policy
        self.ens_size = self.policy.model.num_nets

    def forward_multiple(self, A, S, mean=True, disagr=True):
        ## Takes a list of actions A and a list of states S we want to query the model from
        ## Returns a list of the return of a forward call for each couple (action, state)
        assert len(A) == len(S)
        batch_len = len(A)
        # ens_size = self._dynamics_model.ensemble_size

        proc_s_0 = tf.convert_to_tensor(S, dtype='float32')

        proc_s_0 = self.policy.obs_preproc(proc_s_0)
        
        # s_0, a_0 = self._expand_to_ts_format(s_0), self._expand_to_ts_format(a_0)

        inputs = tf.concat([proc_s_0, A], axis=-1)
        # inputs = tf.cast(inputs, 'float32')
        mean, var  = self.policy.model.create_prediction_tensors(inputs, factored=True, return_numpy=True)

        predictions = self.policy.obs_postproc(S, mean)
        
        mean = np.swapaxes(mean, 0, 1)
        var = np.swapaxes(var, 0, 1)
        predictions = np.swapaxes(predictions, 0, 1)
        
        # return predictions, [0.]*len(predictions)
        return predictions, var
    
        S_0 = np.empty((batch_len*self.ens_size, S.shape[1]))
        A_0 = np.empty((batch_len*self.ens_size, A.shape[1]))

        batch_cpt = 0
        for a, s in zip(A, S):
            S_0[batch_cpt*self.ens_size:batch_cpt*self.ens_size+self.ens_size,:] = \
            np.tile(s,(self.ens_size, 1))
            # np.tile(copy.deepcopy(s),(self._dynamics_model.ensemble_size, 1))

            A_0[batch_cpt*self.ens_size:batch_cpt*self.ens_size+self.ens_size,:] = \
            np.tile(a,(self.ens_size, 1))
            # np.tile(copy.deepcopy(a),(self._dynamics_model.ensemble_size, 1))
            batch_cpt += 1
        return self.forward(A_0, S_0, mean=mean, disagr=disagr, multiple=True)

        return batch_pred_delta_ns, batch_disagreement

    def forward(self, a, s, mean=True, disagr=True, multiple=False):
        s_0 = copy.deepcopy(s)
        a_0 = copy.deepcopy(a)

        if not multiple:
            s_0 = np.tile(s_0,(self.ens_size, 1))
            a_0 = np.tile(a_0,(self.ens_size, 1))

        # s_0 = ptu.from_numpy(s_0)
        # a_0 = ptu.from_numpy(a_0)

        # a_0 = a_0.repeat(self._dynamics_model.ensemble_size,1)

        # if probalistic dynamics model - choose output mean or sample            
        # if disagr:
        #     if not multiple:
        #         pred_delta_ns, disagreement = self._dynamics_model.sample_with_disagreement(
        #             torch.cat((
        #                 self.policy._expand_to_ts_format(s_0),
        #                 self.policy._expand_to_ts_format(a_0)), dim=-1
        #             ), disagreement_type="mean" if mean else "var")
        #         pred_delta_ns = ptu.get_numpy(pred_delta_ns)
        #         return pred_delta_ns, disagreement
        #     else:
        #         pred_delta_ns_list, disagreement_list = \
        #         self._dynamics_model.sample_with_disagreement_multiple(
        #             torch.cat((
        #                 self.policy._expand_to_ts_format(s_0),
        #                 self.policy._expand_to_ts_format(a_0)), dim=-1
        #             ), disagreement_type="mean" if mean else "var")
        #         for i in range(len(pred_delta_ns_list)):
        #             pred_delta_ns_list[i] = ptu.get_numpy(pred_delta_ns_list[i])
        #         return pred_delta_ns_list, disagreement_list
        # else:
        #     pred_delta_ns = self._dynamics_model.output_pred_ts_ensemble(s_0, a_0, mean=mean)

        ## For particle: need 3d inputs: (ens_size, batch_size, input_dim)
        ## Will return (ens_size, batch_size, output_dim)

        proc_s_0 = tf.convert_to_tensor(s_0, dtype='float32')

        proc_s_0 = self.policy.obs_preproc(proc_s_0)
        
        # s_0, a_0 = self._expand_to_ts_format(s_0), self._expand_to_ts_format(a_0)

        inputs = tf.concat([proc_s_0, a_0], axis=-1)
        # inputs = tf.cast(inputs, 'float32')
        pred_delta_ns = self.policy.model.create_prediction_tensors(inputs, factored=True)

        self.policy.obs_postproc(s_0, pred_delta_ns)
        return pred_delta_ns, 0

    def _expand_to_ts_format(self, mat):
        dim = mat.get_shape()[-1]
        return tf.reshape(
            tf.transpose(
                tf.reshape(mat, [-1, self.policy.model.num_nets, self.policy.npart // self.policy.model.num_nets, dim]),
                [1, 0, 2, 3]
            ),
            [self.policy.model.num_nets, -1, dim]
        )

################################################################################
################################# Functions ####################################
################################################################################
def transitions_to_samples(trajs, samples):
    for traj in trajs:
        sample = {}
        sample["obs"] = []
        sample["ac"] = []
        sample["rewards"] = []
        for transition in traj:
            sample["ac"].append(transition[0])
            sample["obs"].append(transition[1])
            sample["rewards"].append(transition[2])

        sample["obs"] = np.array(sample["obs"][:])
        sample["ac"] = np.array(sample["ac"][:-1])
        sample["rewards"] = np.array(sample["rewards"][:])
        samples.append(sample)

    return samples

################################################################################
#################################### Main ######################################
################################################################################
def main(env, ctrl_type, ctrl_args, overrides, logdir, init_method,
         init_episode, num_cores, action_sample_budget, state_sample_budget):
    ctrl_args = DotMap(**{key: val for (key, val) in ctrl_args})
    cfg = create_config(env, ctrl_type, ctrl_args, overrides, logdir)
    cfg.pprint()

    kwargs = {}
    if init_method is not None:
        ## Visualizing method
        from model_init_study.visualization.test_trajectories_visualization \
        import TestTrajectoriesVisualization
        from model_init_study.visualization.n_step_error_visualization \
        import NStepErrorVisualization
        from model_init_study.visualization.dynamics_visualization \
        import DynamicsVisualization

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
        dynamics_model_params = \
        {
            'obs_dim': obs_dim,
            'action_dim': act_dim,
        }
        params = \
        {
            'obs_dim': obs_dim,
            'action_dim': act_dim,

            'n_init_episodes': init_episode,
            'n_test_episodes': int(.2*init_episode),
            'separator': None,
            
            'inc_rew': True,
            # 'controller_type': NeuralNetworkController,
            # 'controller_params': controller_params,
            'dynamics_model_params': dynamics_model_params,
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
            'env_name': env,
            'env_max_h': cfg.exp_cfg.sim_cfg.task_hor,

            ## Dynamics visualizer specific params
            'sample_hor': 1,
            'action_sample_budget': action_sample_budget,
            'state_sample_budget': state_sample_budget,
            'num_cores': num_cores,

            ## dump
            'dump_path': logdir,
        }

        ## Instanciate the initializer
        initializer = Initializer(params)

        kwargs['initializer'] = initializer
    else:
        raise ValueError("No Initialization method was passed to the script as CLI argument.")

    plan_h = 0
    if ctrl_type == "MPC":
        # cfg.exp_cfg.exp_cfg.policy = MPC(cfg.ctrl_cfg)
        policy = MPC(cfg.ctrl_cfg)
        plan_h = policy.plan_hor
        
    ## Instanciate the initializer
    initializer = Initializer(params)

    ## Execute the initializer policies on the environment
    all_trajs = initializer.run()

    ## Separate training and test
    train_trajs = all_trajs[:-params['n_test_episodes']]
    test_trajs = all_trajs[-params['n_test_episodes']:]

    ## Format train actions and trajectories FOR DUMP
    # Actions
    train_actions = np.empty((params['n_init_episodes'],
                              params['env_max_h'],
                              act_dim))
    train_actions[:] = np.nan

    for i in range(params['n_init_episodes']):
        traj_len = params['env_max_h'] if params['env_max_h'] < len(train_trajs[i]) \
                   else len(train_trajs[i])
        for j in range(traj_len):
            train_actions[i, j, :] = train_trajs[i][j][0]
    # Trajectories
    train_trajectories = np.empty((params['n_init_episodes'],
                                   params['env_max_h'],
                                   obs_dim))
    train_trajectories[:] = np.nan

    for i in range(params['n_init_episodes']):
        traj_len = params['env_max_h'] if params['env_max_h'] < len(train_trajs[i]) \
                   else len(train_trajs[i])
        for j in range(traj_len):
            train_trajectories[i, j, :] = train_trajs[i][j][1]

    ## Format test trajectories FOR DUMP
    # Trajectories
    test_trajectories = np.empty((params['n_test_episodes'],
                                  params['env_max_h'],
                                  obs_dim))
    test_trajectories[:] = np.nan

    for i in range(params['n_test_episodes']):
        traj_len = params['env_max_h'] if params['env_max_h'] < len(test_trajs[i]) \
                   else len(test_trajs[i])
        for j in range(traj_len):
            test_trajectories[i, j, :] = test_trajs[i][j][1]
    # Actions
    test_actions = np.empty((params['n_test_episodes'],
                                  params['env_max_h'],
                                  act_dim))
    test_actions[:] = np.nan

    for i in range(params['n_test_episodes']):
        traj_len = params['env_max_h'] if params['env_max_h'] < len(test_trajs[i]) \
                   else len(test_trajs[i])
        for j in range(traj_len):
            test_actions[i, j, :] = test_trajs[i][j][0]

            
    ## Format the training and test samples FOR TRAIN
    train_samples = []
    test_samples = []
    
    train_samples = transitions_to_samples(train_trajs, train_samples)
    test_samples = transitions_to_samples(test_trajs, test_samples)
    ## Train the model (use train policy ?)
    policy.train(
        [sample["obs"] for sample in train_samples],
        [sample["ac"] for sample in train_samples],
        [sample["rewards"] for sample in train_samples]
    )

    ## Get the expert trajectories

    ## We get them from random actions folder results on each env:
    home_path = os.path.expanduser('~')
    path_to_results = ''
    if args.pets_results_path is not None:
        path_to_results = args.pets_results_path
    else:
        path_to_results = os.path.join(home_path, 'mis_results/')

    path_to_test_trajectories = '{}_pets_results/{}_random-actions_pets_results'.format(env, env)

    abs_path_to_trajs = os.path.join(path_to_results, path_to_test_trajectories)

    if env == 'cartpole': 
        ## Will have to do it with os.walk for other envs
        traj1_data = loadmat(os.path.join(abs_path_to_trajs, '1/2023-04-23--15:12:46/logs.mat'))
        traj2_data = loadmat(os.path.join(abs_path_to_trajs, '5/2023-04-23--15:12:46/logs.mat'))
    elif env == 'pusher':
        traj1_data = loadmat(os.path.join(abs_path_to_trajs, '1/2023-04-22--08:37:34/logs.mat'))
        traj2_data = loadmat(os.path.join(abs_path_to_trajs, '5/2023-04-22--08:37:34/logs.mat'))
    elif env == 'reacher':
        traj1_data = loadmat(os.path.join(abs_path_to_trajs, '1/2023-04-20--15:30:24/logs.mat'))
        traj2_data = loadmat(os.path.join(abs_path_to_trajs, '5/2023-04-20--15:30:25/logs.mat'))

        
    def get_test_data(*args):
        ret_acs = []
        ret_trajs = []
        ret_returns = []
        ret_rewards = []
        for traj_data in args:
            ## -1 to get the final traj (~ best)
            ret_acs.append(traj_data["actions"][-1])
            ret_trajs.append(traj_data["observations"][-1])
            ret_returns.append(traj_data["returns"][-1])
            ret_rewards.append(traj_data["rewards"][-1])
        return np.array(ret_acs), np.array(ret_trajs), np.array(ret_returns), np.array(ret_rewards)

    ret_acs, ret_trajs, ret_returns, ret_rewards = get_test_data(traj1_data, traj2_data)

    ## Fix sequences lengths
    ret_acs = ret_acs[:,:-1]; ret_trajs = ret_trajs[:,:-1];
    ret_returns = ret_returns[:,:-1]; ret_rewards[:,:-1];
    
    dynamics_visualizer = DynamicsVisualization(params)
    dynamics_visualizer.dump_plots(0)

    if args.dynamics_only:
        exit(0)

    dynamics_model = WrappedPETSDynamicsModel(policy)
        
    ## Execute each visualize routines
    params['model'] = dynamics_model # to pass down to the visualizer routines

    test_traj_visualizer = TestTrajectoriesVisualization(params)
    test_traj_visualizer.set_test_trajectories(ret_trajs)
    test_traj_visualizer.set_controller(None, actions_lists=ret_acs,
                                     ctrl_type='actions_list', ctrl_input='time')


    n_step_visualizer = NStepErrorVisualization(params)
    n_step_visualizer.set_test_trajectories(ret_trajs)
    n_step_visualizer.set_controller(None, actions_lists=ret_acs,
                                     ctrl_type='actions_list', ctrl_input='time')

    ## FROM EXAMPLE TRAJECTORIES (end trajs that solve the task)
    ## Visualize n step error and disagreement ###

    n_step_visualizer.set_n(1)

    examples_1_step_trajs, examples_1_step_disagrs, examples_1_step_pred_errors = n_step_visualizer.dump_plots(
        env,
        args.init_method,
        init_episode,
        'examples', dump_separate=True, no_sep=True)

    # n_step_visualizer.set_n(plan_h)
    
    # examples_plan_h_step_trajs, examples_plan_h_step_disagrs, examples_plan_h_step_pred_errors = n_step_visualizer.dump_plots(
    #     env,
    #     args.init_method,
    #     init_episode,
    #     'examples', dump_separate=True, no_sep=True)

    ### Full recursive prediction visualizations ###
    examples_pred_trajs, examples_disagrs, examples_pred_errors = test_traj_visualizer.dump_plots(
        env,
        args.init_method,
        args.init_episode,
        'examples', dump_separate=True, no_sep=True)

    ### Full recursive prediction visualizations ###
    test_traj_visualizer.set_test_trajectories(test_trajectories)
    test_pred_trajs, test_disagrs, test_pred_errors = test_traj_visualizer.dump_plots(
        env,
        args.init_method,
        args.init_episode,
        'test', dump_separate=True, no_sep=True)

    data_path = os.path.join(
        logdir,
        '{}_{}_{}_data.npz'.format(env, init_method, init_episode))

    
    np.savez(data_path,
             test_pred_trajs=test_pred_trajs,
             test_disagrs=test_disagrs,
             test_pred_errors=test_pred_errors,
             examples_pred_trajs=examples_pred_trajs,
             examples_disagrs=examples_disagrs,
             examples_pred_errors=examples_pred_errors,
             examples_1_step_trajs=examples_1_step_trajs,
             examples_1_step_disagrs=examples_1_step_disagrs,
             examples_1_step_pred_errors=examples_1_step_pred_errors,
             # examples_plan_h_step_trajs=examples_plan_h_step_trajs,
             # examples_plan_h_step_disagrs=examples_plan_h_step_disagrs,
             # examples_plan_h_step_pred_errors=examples_plan_h_step_pred_errors,
             train_trajs=train_trajectories,
             train_actions=train_actions,
             test_trajs=test_trajectories,
             test_actions=test_actions,)

    print('Finished cleanly for {} environment with {} init method and {} episodes budget'.format(env, args.init_method, args.init_episode))
    print('\n###############################################################################\n')
    exit(0)
    
if __name__ == "__main__":
    import logging
    import os

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
    logging.getLogger('tensorflow').setLevel(logging.FATAL)

    import argparse
    import pprint

    from dotmap import DotMap

    from dmbrl.misc.MBExp import MBExperiment
    from dmbrl.controllers.MPC import MPC
    from dmbrl.config import create_config

    import numpy as np
    import tensorflow as tf

    import copy
    from scipy.io import loadmat

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
    parser.add_argument('--init-episode', type=int, default=10,
                        help='Budget for initial data gathering method')
    parser.add_argument('--pets-results-path', type=str, default=None,
                        help='Path of previous PETS results for running the MIS')
    parser.add_argument('--num-cores', type=int, default=10,
                        help='number of cores to use for parallel computation')
    parser.add_argument('--action-sample-budget', type=int, default=1000,
                        help='Number of different action to sample')
    parser.add_argument('--state-sample-budget', type=int, default=1000,
                        help='Number of states to sample per action')
    parser.add_argument('--dynamics-only', action="store_true",
                        help='If on, will only compute dynamics uniformity')

    args = parser.parse_args()

    main(args.env, "MPC", args.ctrl_arg, args.override, args.logdir, args.init_method, args.init_episode, args.num_cores, args.action_sample_budget, args.state_sample_budget)

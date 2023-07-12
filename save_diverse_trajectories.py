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

def pd_read_csv_fast(filename):
    ## Only read the first line to get the columns
    data = pd.read_csv(filename, nrows=1)
    ## Only keep important columns and 5 genotype columns for merge purposes
    usecols = [col for col in data.columns if 'bd' in col or 'fit' in col]
    usecols += [col for col in data.columns if 'x' in col][:5]
    ## Return the complete dataset (without the << useless >> columns
    return pd.read_csv(filename, usecols=usecols)

def get_closest_to_clusters_centroid(data, bd_cols, n_clusters):
    # Use k-means clustering to divide the data space into n_clusters clusters
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(data[bd_cols])
    
    # Select closest data point from each cluster
    data_centers = pd.DataFrame(columns=data.columns)

    for i in range(n_clusters):
        cluster_center = kmeans.cluster_centers_[i]
        cluster = data[kmeans.labels_ == i]
        # cluster = data[data['cluster'] == i]
        # Calculate the distance of each point in the cluster to the cluster center
        dist = 0
        for j in range(len(bd_cols)):
            dist += (cluster[bd_cols[j]] - cluster_center[j])**2
        cluster['distance'] = dist**0.5
        # Select the point with the minimum distance to the cluster center
        closest_point = cluster.loc[cluster['distance'].idxmin()]
        data_centers = data_centers.append(closest_point)

    return data_centers

def normalize_inputs_o_minmax(data, obs_min, obs_max):
        data_norm = (data - obs_min)/(obs_max - obs_min)
        rescaled_data_norm = data_norm * (1 + 1) - 1 ## Rescale between -1 and 1
        return rescaled_data_norm

def main(args):
    n_waypoints = 2
    # get env info
    obs_dim, act_dim, dim_map, max_step, obs_min, obs_max = process_env(args)
    bd_cols = ['bd'+str(i) for i in range(dim_map*n_waypoints)]
    ## Load data
    rel_fp = args.environment + '_ns_results'
    rel_fp = os.path.join(rel_fp, 'ffnn_2l_10n_2wps_results')

    xs_per_reps = []
    bds_per_reps = []
    for i in range(1, args.n_reps+1):
        rel_fp = os.path.join(rel_fp, str(i))
        filename = os.path.join(rel_fp, 'archive_100100.dat')
        # filename = os.path.join(rel_fp, 'archive_100100_all_evals.dat')

        ## Load it as a pandas dataframe
        # archive_data = pd_read_csv_fast(filename)
        archive_data = pd.read_csv(filename)
        archive_data = archive_data.iloc[:,:-1]

        ## select sel_size individuals closest to sel_size kmeans clusters
        ret_data = get_closest_to_clusters_centroid(archive_data, bd_cols, 10)
        gen_cols = [col for col in ret_data.columns if 'x' in col]

        xs_per_reps.append(ret_data[gen_cols].to_numpy())
        bds_per_reps.append(ret_data[bd_cols].to_numpy())
        
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
    
    controller = NeuralNetworkController(params)
    for i in range(args.n_reps):
        for (x, bd) in zip(xs_per_reps[i], bds_per_reps[i]):
            controller.set_parameters(x)
            obs = gym_env.reset()
            obs_trajs.append([])
            # for j in range(2):
            for t in range(max_step):
                norm_obs = normalize_inputs_o_minmax(obs, obs_min, obs_max)
                a = controller(norm_obs)
                a = np.clip(a, -1, 1)
                obs_trajs[-1].append(obs)
                obs, r, d, info = gym_env.step(a)

            obs_trajs[-1].append(obs)
            wp_idxs = [i for i in range(len(obs_trajs[-1])//n_waypoints, len(obs_trajs[-1]),
                                        len(obs_trajs[-1])//n_waypoints)][:n_waypoints-1]
            wp_idxs += [-1]

            obs_wps = np.take(obs_trajs[-1], wp_idxs, axis=0)

            if args.environment == 'ball_in_cup':
                replay_bd = obs_wps[:,:3].flatten()
            elif args.environment == 'cartpole':
                ## BD is pos of cart + orientation of pole
                replay_bd = obs_wps[:,[0,2]].flatten()
            elif args.environment == 'pusher':
                ## BD is EE pos + object pos
                replay_bd = obs_wps[:,-6:].flatten()
            elif args.environment == 'reacher':
                ## BD is EE pos
                ees = gym_env.get_EE_pos(obs_wps)
                replay_bd = ees.flatten()
            print('bd:        ',bd)
            print('replay bd: ',replay_bd)
        import pdb; pdb.set_trace()

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

    parser.add_argument('--ns-examples-path', type=str, default=None,
                        required=True,
                        help='Path of NS results from which we want to gather' \
                        ' trajectories')

    parser.add_argument('--n-reps', type=int, default=1,
                        help='Number of repetitions of ns exps')
    args = parser.parse_args()

    main(args)

### Inspired from plotter.ipynb
# ps: I hate jupyter notebook
# Author: Elias Hanna



def main(args):
    ## handles args
    init_methods = args.init_methods
    n_reps = args.n_reps
    env_name = args.environment
    min_num_trials = 0
    if env_name == 'reacher':
        min_num_trials = 100
    if env_name == 'pusher':
        min_num_trials = 100
    if env_name == 'cartpole':
        min_num_trials = 50
    if env_name == 'halfcheetah':
        min_num_trials = 300
    if env_name == 'ball_in_cup':
        min_num_trials = 100

    im_returns = np.empty((len(init_methods), n_reps, min_num_trials))
    im_returns[:] = np.nan
    # Get current working dir (folder from which py script was executed)
    root_wd = os.getcwd()
    
    ## Go over init methods
    for (init_method, im_cpt) in zip(init_methods, range(len(init_methods))):
        print(f"Processing {init_method} results on {env_name}...")

        method_wd = os.path.join(root_wd,
                                 f'{env_name}_{init_method}_pets_results/') 
        # get rep dirs (from method_wd pov)
        try:
            rep_dirs = next(os.walk(method_wd))[1]
        except:
            import pdb; pdb.set_trace()
        # switch from method_wd pov to abs pov
        rep_dirs = [os.path.join(method_wd, rep_dir)
                    for rep_dir in rep_dirs]

        # returns = []
        returns = np.empty((n_reps, min_num_trials))
        returns[:] = np.nan
        rep_cpt = 0
        for rep_dir in rep_dirs:
            # get time dirs (from rep_dir pov)
            try:
                time_dirs = next(os.walk(rep_dir))[1]
            except:
                import pdb; pdb.set_trace()
            # switch from method_wd pov to abs pov
            time_dirs = [os.path.join(rep_dir, time_dir)
                        for time_dir in time_dirs]

            # data = loadmat(os.path.join(rep_dir, subdir, "logs.mat"))
            time_dir = time_dirs[0]
            data = loadmat(os.path.join(time_dir, "logs.mat"))
            if data["returns"].shape[1] >= min_num_trials:
                ## Just to remove outliers from CNRW_0...
                if env_name == 'reacher' and '0' in init_method:
                    if data["returns"][0][:min_num_trials][0] < -250:
                        continue
                returns[rep_cpt] = data["returns"][0][:min_num_trials]
                if env_name == 'ball_in_cup':
                    returns[rep_cpt] -= 300
                
            rep_cpt += 1
            
        returns = np.array(returns)
        returns = np.maximum.accumulate(returns, axis=-1)
        im_returns[im_cpt,:,:] = returns

    ## Boxplot of return on first iteration only 
    fig, ax = plt.subplots()
    ## Add to the coverage boxplot the policy search method
    init_returns = im_returns[:,:,0]
    mask = ~np.isnan(init_returns)
    filtered_init_returns = [d[m] for d, m in zip(init_returns, mask)]
    ax.boxplot(filtered_init_returns, 0, '') # don't show the outliers
    # ax.boxplot(im_returns[:,:,0].T, 0, '') # don't show the outliers
    # ax.boxplot(all_psm_covs)
    ax.set_xticklabels(init_methods, fontsize=14)
    ax.tick_params(axis='y', labelsize=12)

    ax.set_ylabel("Return", fontsize=28)

    # plt.title(f"PETS initial return for different model bootstraps on {args.environment} environment")
    fig.set_size_inches(14, 7)
    plt.savefig(f"{args.environment}_bp_return", bbox_inches='tight')


    fig, ax = plt.subplots()

    ## Compute median (excluding nans) for all init_methods 
    returns_medians = np.nanmedian(im_returns, axis=1)
    returns_1qs = np.nanquantile(im_returns, 1/4, axis=1)
    returns_3qs = np.nanquantile(im_returns, 3/4, axis=1)

    trials = np.arange(1, min_num_trials+1)
    for (init_method, returns_median, returns_1q, returns_3q) in zip(init_methods, returns_medians, returns_1qs, returns_3qs):
        ax.plot(trials, returns_median,
                label=init_method)
        ax.fill_between(trials,
                        returns_1q,
                        returns_3q,
                        alpha=0.2)
    # Plot result
    plt.title(f"Impact on PETS performance of different model bootstraps on {env_name}")
    plt.xlabel("Iteration number")
    plt.ylabel("Return")
    fig.set_size_inches(35, 14)
    plt.legend(prop={'size': 20})
    plt.savefig(f"{env_name}_performance_vs_trials",
                dpi=300, bbox_inches='tight')
    #plt.show()
    
    
if __name__ == '__main__':
    import argparse
    import os
    import csv

    import numpy as np
    from scipy.io import loadmat, savemat
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()

    parser.add_argument('--init-methods', nargs="*", type=str,
                        default=['random-actions','brownian-motion',
                                 'colored-noise-beta-0', 'colored-noise-beta-1',
                                 'colored-noise-beta-2'],
                        help='List of initial data gathering methods')
    parser.add_argument('--n-reps', type=int, default=10)
    parser.add_argument('--environment','-e', type=str, default='pusher')

    ## Process CL args
    args = parser.parse_args()

    main(args)

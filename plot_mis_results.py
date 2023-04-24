### Inspired from plotter.ipynb
# ps: I hate jupyter notebook
# Author: Elias Hanna



def main(args):
    ## handles args
    init_methods = args.init_methods
    n_reps = args.n_reps
    returns = np.array((len(init_methods), n_reps))
    # Get current working dir (folder from which py script was executed)
    root_wd = os.getcwd()
    

    
    returns = []
    for subdir in os.listdir(log_dir):
        data = loadmat(os.path.join(log_dir, subdir, "logs.mat"))
        if data["returns"].shape[1] >= min_num_trials:
            returns.append(data["returns"][0][:min_num_trials])

    returns = np.array(returns)
    returns = np.maximum.accumulate(returns, axis=-1)
    mean = np.mean(returns, axis=0)

    # Plot result
    plt.figure()
    plt.plot(np.arange(1, min_num_trials + 1), mean)
    plt.title("Performance")
    plt.xlabel("Iteration number")
    plt.ylabel("Return")
    plt.show()

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
    parser.add_argument('-e', type=str, default='pusher')

    ## Process CL args
    args = parser.parse_args()

    main(args)

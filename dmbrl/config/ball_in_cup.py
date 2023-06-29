from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import tensorflow as tf
from dotmap import DotMap
import gym

from dmbrl.misc.DotmapUtils import get_required_argument
from dmbrl.modeling.layers import FC
import mb_ge

class BallInCup3dConfigModule:
    ENV_NAME           = 'BallInCup3d-v0'
    TASK_HORIZON       = 300
    NTRAIN_ITERS       = 100
    NROLLOUTS_PER_ITER = 1
    PLAN_HOR           = 25
    MODEL_IN, MODEL_OUT = 9, 6
    GP_NINDUCING_POINTS = 200

    def __init__(self):
        self.ENV = gym.make(self.ENV_NAME)
        self.ENV.dense = True
        cfg = tf.ConfigProto()
        cfg.gpu_options.allow_growth = True
        cfg.gpu_options.per_process_gpu_memory_fraction = 0.1
        cfg.log_device_placement = True
        self.SESS = tf.Session(config=cfg)
        # self.NN_TRAIN_CFG = {"epochs": 5}
        self.NN_TRAIN_CFG = {"epochs": 20}
        self.OPT_CFG = {
            "Random": {
                "popsize": 2000,
            },
            # "CEM": {
            #     "popsize":    400,
            #     "num_elites": 40,
            #     "max_iters":  5,
            #     "alpha":      0.1
            # }
            "CEM": {
                "popsize":    50,
                "num_elites": 5,
                "max_iters":  5,
                "alpha":      0.1
            }
        }
        self.UPDATE_FNS = []

        # Fill in other things to be done here.

    @staticmethod
    def obs_preproc(obs):
        # # Note: Must be able to process both NumPy and Tensorflow arrays.
        # if isinstance(obs, np.ndarray):
        #     raise NotImplementedError()
        # else:
        #     raise NotImplementedError
        return obs
    
    @staticmethod
    def obs_postproc(obs, pred):
        # # Note: Must be able to process both NumPy and Tensorflow arrays.
        # if isinstance(obs, np.ndarray):
        #     raise NotImplementedError()
        # else:
        #     raise NotImplementedError()
        return obs + pred
    
    @staticmethod
    def targ_proc(obs, next_obs):
        # # Note: Only needs to process NumPy arrays.
        # raise NotImplementedError()
        return next_obs - obs
        
    @staticmethod
    def obs_cost_fn(obs):
        obs_costs = None
        if isinstance(obs, np.ndarray):
            ## This reward function creates an empty reward zone right below the cup
            # abs_obs = np.abs(obs)
            # obs_costs = np.linalg.norm(obs[:,:3], axis=1)
            # zero_inds = np.where((abs_obs[:,0] < 0.1 )
            #                      & (abs_obs[:,1] < 0.1)
            #                      & (abs_obs[:,2] > 0)
            #                      & (abs_obs[:,2] < 0.3))
            # obs_costs[zero_inds] = 0.0

            ## This reward function rewards policies lifting the ball while
            ## mainting the rope tense as long as the ball is below the cup
            ## If the ball goes above the cup, reward switches to be maximal
            ## when the distance between the ball and the cup is minimal
            tense_rew = -np.linalg.norm(obs[:,:3], axis=1)/.358
            lifting_rew = -(obs[:,2] - .358)/(0 - .358)
            
            below_rew = tense_rew + lifting_rew
            
            target_close_rew = -2 -2*(np.linalg.norm(obs[:,:3]) - .358)/(0 - .358)

            above_rew = target_close_rew

            obs_costs = np.empty(below_rew.shape)
            obs_costs[obs[:,3] >= 0] = below_rew[obs[:,3] >= 0]
            obs_costs[obs[:,3] < 0] = above_rew[obs[:,3] < 0]

        else:
            ## This reward function creates an empty reward zone right below the cup
            # obs_costs = tf.norm(obs[:,:3], axis=1)
            # abs_obs = tf.abs(obs)
            # cond_1 = tf.less_equal(abs_obs[:,0], tf.constant(0.1))
            # # obs_costs.assign(tf.where(cond_1,
            # #                              tf.zeros_like(obs_costs), obs_costs))
            # cond_2 = tf.less_equal(abs_obs[:,1], tf.constant(0.1))
            # # obs_costs.assign(tf.where(cond_2,
            # #                              tf.zeros_like(obs_costs), obs_costs))
            # cond_3 = tf.logical_and(cond_1, cond_2) ## Logical and of 1 & 2
            # cond_4 = tf.greater(obs[:,2], tf.constant(0.0)) ## ball is below cup
            # cond_5 = tf.logical_and(cond_3, cond_4)
            # cond_6 = tf.less(abs_obs[:,2], tf.constant(0.3))
            # cond_7 = tf.logical_and(cond_5, cond_6)
            # obs_costs = tf.where(cond_7,
            #                      tf.zeros_like(obs_costs), obs_costs)

            tense_rew = -tf.divide(tf.norm(obs[:,:3], axis=1), tf.constant(.358))
            lifting_rew = -tf.divide((tf.subtract(obs[:,2], tf.constant(.358))),
                                     tf.constant(.358))
            speed_rew = tf.multiply(obs[:,-1], tf.constant(2.))

            below_rew = tf.add(tense_rew, lifting_rew)
            # below_rew = tf.add(tense_rew, speed_rew)
            
            target_close_rew = tf.subtract(tf.norm(obs[:,:3], axis=1), tf.constant(.358))
            target_close_rew = tf.divide(target_close_rew, tf.constant(-.358))
            target_close_rew = -tf.multiply(tf.constant(2.), target_close_rew)
            above_rew = tf.add(tf.constant(-2.), target_close_rew)

            cond = tf.greater(obs[:,2], tf.constant(0.0)) ## ball is below cup

            obs_costs = tf.where(cond, below_rew, above_rew)

            # abs_obs = tf.abs(obs)
            # cond_in_1 = tf.less_equal(abs_obs[:,0], tf.constant(0.042))
            # cond_in_2 = tf.less_equal(abs_obs[:,1], tf.constant(0.042))
            # cond_in_3 = tf.less_equal(abs_obs[:,2], tf.constant(0.042))
            # cond_in = tf.logical_and(cond_in_1, cond_in_2)
            # cond_in = tf.logical_and(cond_in, cond_in_3)

            # obs_costs = tf.where(cond_in, tf.constant(1000., shape=(20,)), obs_costs)
            
        return obs_costs
            
    @staticmethod
    def ac_cost_fn(acs):
        # # Note: Must be able to process both NumPy and Tensorflow arrays.
        # if isinstance(acs, np.ndarray):
        #     raise NotImplementedError()
        # else:
        #     raise NotImplementedError()
        if isinstance(acs, np.ndarray):
            return 0.01 * np.sum(np.square(acs), axis=1)
        else:
            return 0.01 * tf.reduce_sum(tf.square(acs), axis=1)

    def nn_constructor(self, model_init_cfg):
        model = get_required_argument(model_init_cfg, "model_class", "Must provide model class")(DotMap(
            name="model", num_networks=get_required_argument(model_init_cfg, "num_nets", "Must provide ensemble size"),
            sess=self.SESS, load_model=model_init_cfg.get("load_model", False),
            model_dir=model_init_cfg.get("model_dir", None)
        ))
        # Construct model below. For example:
        # model.add(FC(*args))
        # ...
        # model.finalize(tf.train.AdamOptimizer, {"learning_rate": 0.001})
        if not model_init_cfg.get("load_model", False):
            model.add(FC(500, input_dim=self.MODEL_IN, activation='swish', weight_decay=0.0001))
            model.add(FC(500, activation='swish', weight_decay=0.00025))
            model.add(FC(500, activation='swish', weight_decay=0.00025))
            model.add(FC(self.MODEL_OUT, weight_decay=0.0005))
        model.finalize(tf.train.AdamOptimizer, {"learning_rate": 0.001})
        return model


CONFIG_MODULE = BallInCup3dConfigModule


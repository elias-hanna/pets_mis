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
        cfg = tf.ConfigProto()
        cfg.gpu_options.allow_growth = True
        self.SESS = tf.Session(config=cfg)
        self.NN_TRAIN_CFG = {"epochs": None}
        self.OPT_CFG = {
            "Random": {
                "popsize": None
            },
            "CEM": {
                "popsize":    None,
                "num_elites": None,
                "max_iters":  None,
                "alpha":      None
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
        # Note: Must be able to process both NumPy and Tensorflow arrays.
        ## Define a cylinder below the cup with no reward  
        if abs(obs[0]) < .1 and abs(obs[1]) < .1 and obs[2] > 0:
            return 0
        ## Return distance to target
        else:
            if isinstance(obs, np.ndarray):
                return np.linalg.norm(obs[:3])
            else:
                return tf.norm(obs[:3])
            
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


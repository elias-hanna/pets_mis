from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


class PusherEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        mujoco_env.MujocoEnv.__init__(self, '%s/assets/pusher.xml' % dir_path, 4)
        utils.EzPickle.__init__(self)
        self.reset_model()

    def _step(self, a):
        obj_pos = self.get_body_com("object"),
        vec_1 = obj_pos - self.get_body_com("tips_arm")
        vec_2 = obj_pos - self.get_body_com("goal")

        reward_near = -np.sum(np.abs(vec_1))
        reward_dist = -np.sum(np.abs(vec_2))
        reward_ctrl = -np.square(a).sum()
        reward = 1.25 * reward_dist + 0.1 * reward_ctrl + 0.5 * reward_near

        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, {}

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 4.0

    def reset_model(self):
        qpos = self.init_qpos

        self.goal_pos = np.asarray([0, 0])
        self.cylinder_pos = np.array([-0.25, 0.15]) + np.random.normal(0, 0.025, [2])

        qpos[-4:-2] = self.cylinder_pos
        qpos[-2:] = self.goal_pos
        qvel = self.init_qvel + self.np_random.uniform(low=-0.005,
                high=0.005, size=self.model.nv)
        qvel[-4:] = 0
        self.set_state(qpos, qvel)
        self.ac_goal_pos = self.get_body_com("goal")

        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat[:7],
            self.model.data.qvel.flat[:7],
            self.get_body_com("tips_arm"),
            self.get_body_com("object"),
        ])

    def sample_q_vectors(self):
        ## Need to double check the 5 last qs (and all others once again) 
        # qpos_min = np.array([-np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2,
        #                      -np.pi/2, -np.pi/2, 0, 0, 0, 0])
        # qpos_max = np.array([np.pi/2, np.pi/2, np.pi/2, np.pi/2, np.pi/2,
        #                      np.pi/2, np.pi/2, 0, 0, 0, 0])
        # qvel_min = np.array([-0.1, -0.1, -0.1, -0.1, -0.1,
        #                      -0.1, -0.1, 0, 0, 0, 0])
        # qvel_max = np.array([0.1, 0.1, 0.1, 0.1, 0.1,
        #                      0.1, 0.1, 0, 0, 0, 0])

        qpos_min = np.array([-np.pi, -np.pi, -np.pi, -np.pi, -np.pi,
                             -np.pi, -np.pi, 1, 1, 1, 1])
        qpos_max = np.array([np.pi, np.pi, np.pi, np.pi, np.pi,
                             np.pi, np.pi, 1, 1, 1, 1])
        qvel_min = np.array([-1, -1, -1, -1, -1,
                             -1, -1, -0.1, -0.1, -0.1, -0.1])
        qvel_max = np.array([1, 1, 1, 1, 1,
                             1, 1, 0.1, 0.1, 0.1, 0.1])

        qpos = np.zeros(11)
        qvel = np.zeros(11)
        ## Sample qpos and qvel
        qpos = np.random.uniform(low=qpos_min, high=qpos_max, size=(11,))
        qvel = np.random.uniform(low=qvel_min, high=qvel_max, size=(11,))
        ## Reconstruct state from qpos and qvel
        self.set_state(qpos, qvel)
        sample_state = self._get_obs()
        ## Return qpos, qvel and corresponding state
        return qpos, qvel, sample_state

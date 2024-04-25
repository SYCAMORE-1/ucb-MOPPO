# Ant-v2 env
# two objectives
# x-axis speed, y-axis speed

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from os import path

class AntEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.obj_dim = 2
        self.cost_weights = np.ones(self.obj_dim) / self.obj_dim
        mujoco_env.MujocoEnv.__init__(self, model_path = path.join(path.abspath(path.dirname(__file__)), "assets/ant.xml"), frame_skip = 5)
        utils.EzPickle.__init__(self)

    def step(self, a):
        xposbefore = self.get_body_com("torso")[0]
        yposbefore = self.get_body_com("torso")[1]
        a = np.clip(a, -1.0, 1.0)
        self.do_simulation(a, self.frame_skip)

        xposafter = self.get_body_com("torso")[0]
        yposafter = self.get_body_com("torso")[1]

        ctrl_cost = .5 * np.square(a).sum()
        survive_reward = 1.0
        other_reward = - ctrl_cost + survive_reward

        vx_reward = (xposafter - xposbefore) / self.dt +other_reward
        vy_reward = (yposafter - yposbefore) / self.dt+other_reward

        reward = self.cost_weights[0] * vx_reward + self.cost_weights[1] * vy_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all()
        done = not notdone
        ob = self._get_obs()
        return ob, reward, done, {'obj': np.array([vx_reward, vy_reward])}

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat,
        ])

    def reset_model(self):
        self.set_state(self.init_qpos, self.init_qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def set_params(self, params):
        if params['cost_weights'] is not None:
            self.cost_weights = np.copy(params["cost_weights"])
# import numpy as np
# from gym import Env, spaces
# from gym.envs.registration import register
# from gym.envs.mujoco import mujoco_env

# class AntEnv(mujoco_env.MujocoEnv, utils.EzPickle):
#     def __init__(self, xml_file='ant.xml', frame_skip=5):
#         self.obj_dim = 2
#         self.cost_weights = np.ones(self.obj_dim) / self.obj_dim
#         super(AntEnv, self).__init__(xml_file=xml_file, frame_skip=frame_skip)

#     def step(self, action):
#         action = np.clip(action, -1.0, 1.0)
#         xposbefore = self.get_body_com("torso")[0]
#         yposbefore = self.get_body_com("torso")[1]
#         self.do_simulation(action, self.frame_skip)
#         xposafter = self.get_body_com("torso")[0]
#         yposafter = self.get_body_com("torso")[1]

#         ctrl_cost = .5 * np.square(action).sum()
#         survive_reward = 1.0
#         other_reward = -ctrl_cost + survive_reward

#         vx_reward = (xposafter - xposbefore) / self.dt + other_reward
#         vy_reward = (yposafter - yposbefore) / self.dt + other_reward

#         reward = self.cost_weights[0] * vx_reward + self.cost_weights[1] * vy_reward
#         done = not np.isfinite(self.state_vector()).all()
#         ob = self._get_obs()
#         return ob, reward, done, dict(obj=np.array([vx_reward, vy_reward]))

#     def _get_obs(self):
#         return np.concatenate([
#             self.sim.data.qpos.flat[2:],
#             self.sim.data.qvel.flat,
#         ])

#     def reset_model(self):
#         qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
#         qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
#         self.set_state(qpos, qvel)
#         return self._get_obs()

# # Register the environment
# register(
#     id='CustomAnt-v0',
#     entry_point='path.to.your.AntEnv:AntEnv',
#     max_episode_steps=1000,
#     reward_threshold=6000.0,
# )
